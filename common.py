import os
import sys
from coffea.nanoevents import DelphesSchema
import numpy as np
import numba as nb
from numpy.typing import NDArray
import awkward as ak
import vector as vec
from coffea.nanoevents.methods.delphes import behavior
import matplotlib as mpl
import fnmatch
import shutil
from glob import glob
from XRootD import client as xrootd_client
import pickle

def config_path():
    config_dir = os.path.join(os.getcwd(), "configs")
    sys.path.append(config_dir)
    sys.path.extend([path for path in os.listdir(config_dir) if os.path.isdir(path)])

DelphesSchema.mixins.update({
    "ParticleFlowCandidate": "Particle",
    "DarkPartonCandidate": "Particle",
    "DarkHadronCandidate": "Particle",
    "GenCandidate": "Particle",
    "GenStableCandidate": "Particle",
    "GenParticle": "Particle",
    "FatJet": "Jet",
    "GenFatJet": "Jet",
    "DarkPartonJet": "Jet",
    "DarkHadronJet": "Jet",
    "DarkHadronVisibleJet": "Jet",
    "DarkHadronStableJet": "Jet",
})

# workaround for https://cp3.irmp.ucl.ac.be/projects/delphes/ticket/1170
# manually fix mass units
def fix_delphes_mass_units(events):
    GenParticleCollections = [
        "GenParticle",
        "GenCandidate",
        "GenStableCandidate",
        "DarkPartonCandidate",
        "DarkHadronCandidate",
    ]
    for col in GenParticleCollections:
        events[col, "Mass"] = events[col]["Mass"]*0.001
    return events

class DelphesSchema2(DelphesSchema):
    jet_const_pairs = {
        "FatJet" : "ParticleFlowCandidate",
        "Jet" : "ParticleFlowCandidate",
        "DarkPartonJet" : "DarkPartonCandidate",
        "DarkHadronJet" : "DarkHadronCandidate",
        "DarkHadronVisibleJet": "GenCandidate",
        "DarkHadronStableJet": "GenStableCandidate",
        "GenFatJet" : "GenCandidate",
        "GenJet" : "GenCandidate",
    }

    def __init__(self, base_form):
        entries = list(zip(base_form["fields"], base_form["contents"]))
        # branches from the observables step (prefix F_) are plain jagged arrays without a "<name>_size"
        # counter, which DelphesSchema cannot handle: set them aside here and add them back below
        extra = [entry for entry in entries if entry[0].startswith("F_")]
        # avoid weird error when adding constituents
        # these two lists have to be kept in sync: zip, drop, unzip
        base_form["fields"], base_form["contents"] = zip(*[entry for entry in entries if not "fBits" in entry[0] and not entry[0].startswith("F_")])
        # enable using ak.sum on derived class
        behavior[ak.sum, "Particle"] = vec.backends.awkward.behavior[ak.sum, "Momentum4D"]
        super().__init__(base_form)
        # add the F_ branches back as plain fields of the events
        for field, content in extra:
            self._form["fields"].append(field)
            self._form["contents"].append(content)

# ignore unnecessary warning
from numba.core.errors import NumbaTypeSafetyWarning
import warnings
warnings.simplefilter('ignore',category=NumbaTypeSafetyWarning)
# optimized kernel for jet:constituent matching within an event
@nb.njit("i8[:](i4[:],u4[:])")
def get_constituents_kernel(jet_refs: NDArray[np.int32], cand_ids: NDArray[np.uint32]) -> NDArray[np.int64]:
    # get hash table mapping global index : global unique ID
    hash_table = {k:v for v,k in enumerate(cand_ids)}
    # apply hash map
    output = [hash_table[ref] for ref in jet_refs]
    return np.asarray(output)

# apply kernel to events (in chunks)
def get_constituents_chunk(events, jetsname, candsname):
    jets = events[jetsname]
    cands = events[candsname]

    flat_indices = []
    counts_all = []
    jets_per_event = []

    # compute candidate offsets
    cand_counts = ak.num(cands, axis=1)
    cand_offsets = np.cumsum(np.concatenate([[0], ak.to_numpy(cand_counts[:-1])]))

    for i, (jets_evt, cands_evt) in enumerate(zip(jets, cands)):
        if jets_evt is None:
            jets_per_event.append(0)
            continue

        refs = ak.flatten(jets_evt.Constituents.refs)
        counts = ak.num(jets_evt.Constituents.refs, axis=1)

        jets_per_event.append(len(counts))
        counts_all.extend(counts.tolist())

        if len(refs) == 0:
            flat_indices.append(np.array([], dtype=np.int64))
            continue

        # local indices within event
        local_idx = get_constituents_kernel(
            ak.to_numpy(refs),
            ak.to_numpy(cands_evt.fUniqueID),
        )

        # convert to global indices
        global_idx = local_idx + cand_offsets[i]

        flat_indices.append(global_idx)

    # concatenate indices only (cheap)
    flat_indices = np.concatenate(flat_indices) if flat_indices else np.array([], dtype=np.int64)

    # zero-copy flatten of candidates
    flat_cands = ak.flatten(cands)

    # single gather
    gathered = flat_cands[flat_indices]

    # rebuild structure (one level at a time)
    counts_all = np.asarray(counts_all, dtype=np.int64)
    jets_per_event = np.asarray(jets_per_event, dtype=np.int64)
    jets_level = ak.unflatten(gathered, counts_all)
    events_level = ak.unflatten(jets_level, jets_per_event)

    return ak.with_name(events_level, DelphesSchema2.mixins[candsname])

def get_constituents(events, jetsname, candsname, chunk_size=500):
    outputs = []

    # chunking avoids memory overusage
    for start in range(0, len(events), chunk_size):
        stop = start + chunk_size
        chunk = events[start:stop]

        outputs.append(
            get_constituents_chunk(chunk, jetsname, candsname)
        )

    return ak.with_name(
        ak.concatenate(outputs),
        DelphesSchema2.mixins[candsname]
    )

def init_constituents(events):
    for jet,const in DelphesSchema2.jet_const_pairs.items():
        events[jet,"ConstituentsOrig"] = events[jet,"Constituents"]
        events[jet,"Constituents"] = get_constituents(events,jet,const)
    return events

# helper to test that all jet constituents were found
def sum_4vec(vec):
    summed_vec = {
        "t": ak.sum(vec.energy,axis=-1),
        "x": ak.sum(vec.px,axis=-1),
        "y": ak.sum(vec.py,axis=-1),
        "z": ak.sum(vec.pz,axis=-1),
    }
    return ak.zip(summed_vec,with_name="LorentzVector")

def test_constituents(events):
    for jet in DelphesSchema2.jet_const_pairs:
        check_jets = sum_4vec(events[jet,"Constituents"])
        print(jet, check_jets.mass-events[jet].mass)

def load_sample(sample,helper=None,schema=DelphesSchema,with_constituents=False):
    from coffea.nanoevents import NanoEventsFactory
    path = f'models/{sample["model"]}'
    if helper is None:
        from svjHelper import svjHelper
        sample["helper"] = svjHelper.build(f'{path}/config.py')
    else:
        sample["helper"] = helper
    metadict = sample["helper"].metadata()
    metadict["dataset"] = sample["name"]
    sample["events"] = load_events(f'{path}/events.root',schema=schema,metadict=metadict,with_constituents=with_constituents)

def load_events(filename,schema=DelphesSchema,metadict=None,with_constituents=False):
    from coffea.nanoevents import NanoEventsFactory
    if with_constituents and schema==DelphesSchema:
        schema = DelphesSchema2

    events = NanoEventsFactory.from_root(
        file={filename : "Delphes"},
        schemaclass=schema,
        metadata=metadict,
    ).events()

    events = fix_delphes_mass_units(events)

    if with_constituents:
        events = init_constituents(events)

    return events

def set_plot_style():
    # stylistic options
    mpl.rcParams.update({
        "axes.labelsize" : 18,
        "legend.fontsize" : 16,
        "xtick.labelsize" : 14,
        "ytick.labelsize" : 14,
        "font.size" : 18,
        "legend.frameon": True,
    })
    # based on https://github.com/mpetroff/accessible-color-cycles
    # red, blue, mauve, orange, purple, gray,
    colors = ["#e42536", "#5790fc", "#964a8b", "#f89c20", "#7a21dd", "#9c9ca1"]

    # last two are dashdotdot and dashdashdot
    lines = ["solid", "dashed", "dotted", "dashdot", (0, (3, 5, 1, 5, 1, 5)), (0, (3, 5, 3, 5, 1, 5))]
    markers = ['o', 's', 'D', 'v', '^', '*']
    custom_cycler = mpl.cycler(color=colors) + mpl.cycler(linestyle=lines) + mpl.cycler(marker=markers)
    return custom_cycler

EOS_REDIRECTOR = "root://cmseos.fnal.gov/"

def resolve_models(pattern):
    # logical filenames (/store/...) are read from EOS via xrootd; anything else is a local path
    if not pattern.startswith("/store"):
        return glob(pattern)

    directory, name_pattern = pattern.rsplit("/", 1)
    local_base = directory[directory.index("models"):]

    fs = xrootd_client.FileSystem(EOS_REDIRECTOR)
    status, listing = fs.dirlist(directory)
    if not status.ok:
        raise RuntimeError(f"Couldn't read {directory} on EOS: {status.message}")

    matched_names = [entry.name for entry in listing.dirlist if fnmatch.fnmatch(entry.name, name_pattern)]
    if not matched_names:
        return []

    copy_process = xrootd_client.CopyProcess()
    local_dirs = []
    for name in matched_names:
        remote_dir = f"{directory}/{name}"
        local_dir = f"{local_base}/{name}"
        status, model_listing = fs.dirlist(remote_dir)
        if not status.ok:
            raise RuntimeError(f"Couldn't read {remote_dir} on EOS: {status.message}")
        pkl_names = [entry.name for entry in model_listing.dirlist if entry.name.endswith(".pkl")]
        if not pkl_names:
            raise RuntimeError(f"{remote_dir} doesn't have any .pkl files")
        for fname in pkl_names:
            source = f"{EOS_REDIRECTOR}{remote_dir}/{fname}"
            dest = os.path.abspath(f"{local_dir}/{fname}")
            copy_process.add_job(source, dest, mkdir=True, force=True)
            local_dirs.append(local_dir)
    copy_process.prepare()
    _, results = copy_process.run()

    for local_dir, result in zip(local_dirs, results):
        if not result['status'].ok:
            print(f"xrootd copy failed for {local_dir}: {result['status'].message}")
            shutil.rmtree(local_dir, ignore_errors=True)

    return glob(pattern[pattern.index("models"):])

def accumulate_data(samples):
    data = {} # hists + metadata for all models
    for sample in samples:
        #if sample_list and sample["name"] not in sample_list: continue
        data[sample["name"]] = []
        for model in sample['models']:
            file = f'{model}/Hists.pkl'
            with open(file, "rb") as inp:
                data_model = pickle.load(inp)
            file2 = f'{model}/Meta.pkl'
            with open(file2, "rb") as inp:
                metadict = pickle.load(inp)
            # track filename
            data_model['file'] = file
            # merge dicts
            data_model['model'].update(metadict)
            # for convenience, combine model-based and derived metadata
            data_model['meta'] = data_model['model'] | data_model['analysis']

            data[sample["name"]].append(data_model)
    return data
