import numpy as np
import awkward as ak
import fastjet
import uproot
import vector
from common import load_events

vector.register_awkward()

# value stored for jets that cannot be assigned an observable (too few constituents, etc.)
SENTINEL = -99.0

# branches of the friend tree that get merged into the Delphes tree (used by run_model and passed to merge_trees.C)
FRIEND_BRANCHES = ["FatJet_N2", "FatJet_N3", "FatJet_LundX", "FatJet_LundY"]


def to_f32_jagged(arr):
    # Fill missing whole events with empty lists: None -> []
    arr = ak.fill_none(arr, [], axis=0)

    # Fill missing jet-level values with 0.0: [None] -> [0.0]
    arr = ak.fill_none(arr, 0.0, axis=-1)

    arr = ak.values_astype(arr, np.float64)
    arr = ak.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return ak.values_astype(arr, np.float32)


# Force the number of entries per event in an observable to match the number of FatJets
def force_fatjet_layout(arr, FJ):
    arr = to_f32_jagged(arr)

    # position of each FatJet within its event, e.g. [[0, 1], [], [0], ...]
    jet_index = ak.local_index(FJ, axis=1)

    # pad every event to the longest list, so that every FatJet position exists in every event
    longest = int(max(ak.max(ak.num(arr, axis=1), initial=0), ak.max(ak.num(FJ, axis=1), initial=0)))
    padded = ak.pad_none(arr, longest, axis=1)

    # keep exactly one entry per FatJet (extra values are dropped), and fill the padding with zeros
    fixed = ak.fill_none(padded[jet_index], 0.0)
    return ak.to_packed(ak.values_astype(fixed, np.float32))


def count_constituents(jets):
    # number of constituents of each jet in a flat jet array
    return ak.to_numpy(ak.fill_none(ak.num(jets["Constituents"], axis=1), 0))


def cluster_jets(jets, R):
    # one FastJet cluster sequence per jet, made directly from the (vector) constituents
    jet_def = fastjet.JetDefinition(fastjet.cambridge_algorithm, R)
    return fastjet.ClusterSequence(ak.to_packed(jets["Constituents"]), jet_def)


def builtin_ecf(cs, func, beta):
    # FastJet's built-in ECF variation (e.g. "n2", "n3") of each jet when clustered exclusively into one jet
    values = cs.exclusive_jets_energy_correlator(njets=1, beta=beta, func=func)
    values = ak.fill_none(values, np.nan)
    return ak.to_numpy(ak.flatten(values, axis=None)).astype(np.float64)


def lund_points_per_event(values, n_per_jet, n_per_event):
    """
    Flat list of Lund points -> shape: event -> Lund point (float32).
    n_per_jet: number of points of each jet, n_per_event: number of jets of each event.
    All the points of the jets of an event are put together.
    """
    per_jet = ak.unflatten(values, n_per_jet)  # jet -> point
    per_event = ak.unflatten(per_jet, n_per_event)  # event -> jet -> point
    return to_f32_jagged(ak.flatten(per_event, axis=2))  # event -> point

def calculate_fatjet_observables(FJ, beta=1.0, fill_value=SENTINEL, lund_R=0.8):
    """
    N2, N3 and the primary Lund-plane points of each FatJet, using FastJet.
    All jets are clustered once (Cambridge/Aachen), and all observables are taken from that clustering.

        N2: FastJet's built-in "n2", needs at least 3 constituents
        N3: FastJet's built-in "n3", needs at least 4 constituents
        Lund points: LundX = log(lund_R/Delta), LundY = log(kt), need at least 2 constituents

    Jets that cannot be assigned an N2 or N3 value get fill_value.
    Returns two dictionaries and the number of jets used for the Lund points:
        fatjet_obs: "N2", "N3" (shape: event -> FatJet), written into the FatJet record
        lund_obs: "LundX", "LundY" (shape: event -> Lund point), written as separate records
    """
    jets = ak.flatten(FJ, axis=1)  # all jets of all events in one list
    n_const = count_constituents(jets)

    n2 = np.full(len(jets), fill_value, dtype=np.float64)
    n3 = np.full(len(jets), fill_value, dtype=np.float64)
    n_points = np.zeros(len(jets), dtype=np.int64)  # number of Lund points of each jet
    x_flat = np.zeros(0, dtype=np.float64)
    y_flat = np.zeros(0, dtype=np.float64)
    n_jets = 0

    # Cluster once
    clustered = np.flatnonzero(n_const >= 2)
    if len(clustered) > 0:
        cs = cluster_jets(jets[clustered], R=1000.0)
        nc = n_const[clustered]


        n2_all = builtin_ecf(cs, "n2", beta)
        n3_all = builtin_ecf(cs, "n3", beta)
        ok2 = (nc >= 3) & np.isfinite(n2_all)
        ok3 = (nc >= 4) & np.isfinite(n3_all)
        n2[clustered[ok2]] = n2_all[ok2]
        n3[clustered[ok3]] = n3_all[ok3]

        # Primary Lund declustering
        lund = cs.exclusive_jets_lund_declusterings(njets=1)
        delta = lund["Delta"]
        kt = lund["kt"]

        # make sure there is exactly one list of declusterings per jet
        while delta.ndim > 2:
            delta = ak.flatten(delta, axis=1)
            kt = ak.flatten(kt, axis=1)

        good = np.isfinite(delta) & np.isfinite(kt) & (delta > 0.0) & (kt > 0.0)
        x = np.log(lund_R / delta[good])
        y = np.log(kt[good])

        n_jets = len(clustered)
        n_points[clustered] = ak.to_numpy(ak.num(x, axis=1))
        x_flat = ak.to_numpy(ak.flatten(x, axis=1))
        y_flat = ak.to_numpy(ak.flatten(y, axis=1))

    n_per_event = ak.to_numpy(ak.fill_none(ak.num(FJ, axis=1), 0))

    # back to event -> FatJet
    N2 = ak.to_packed(ak.unflatten(n2.astype(np.float32), n_per_event))
    N3 = ak.to_packed(ak.unflatten(n3.astype(np.float32), n_per_event))

    # shape: event -> Lund point
    LundX = lund_points_per_event(x_flat, n_points, n_per_event)
    LundY = lund_points_per_event(y_flat, n_points, n_per_event)

    # per-FatJet values (event -> FatJet) go into the FatJet record
    fatjet_obs = {"N2": N2, "N3": N3}
    # separate records: the number of Lund points differs from the number of FatJets
    lund_obs = {"LundX": LundX, "LundY": LundY}

    return fatjet_obs, lund_obs, n_jets


def make_friend_tree(root_fname, friend_fname, with_constituents=True, verbose=False):
    """
    Compute the additional dark-sector observables of the FatJets in root_fname
    (N2, N3 and the primary Lund-plane points) and write them to a "friend" tree
    in friend_fname, with exactly one entry per entry of the Delphes tree.
    """
    if not with_constituents:
        raise ValueError("the dark-sector observables require jet constituents")

    events = load_events(root_fname, with_constituents=True)
    n_evt = len(events)
    FJ = events.FatJet

    fatjet_obs, lund_obs, n_jets = calculate_fatjet_observables(FJ, beta=1.0, fill_value=SENTINEL, lund_R=0.8)

    # make sure the friend tree has the same number of entries as the Delphes tree
    for name, arr in {**fatjet_obs, **lund_obs}.items():
        if len(arr) != n_evt:
            raise RuntimeError(
                f"{name} has {len(arr)} events but Delphes has {n_evt}: "
                "do not drop or reorder events when computing friend branches."
            )

    # force all jet-level observables to use the exact FatJet layout
    fatjet_obs = {name: force_fatjet_layout(arr, FJ) for name, arr in fatjet_obs.items()}

    # Zip the jet-level variables into one FatJet record, so there is only one counter
    # per group structure (event -> jet, and event -> Lund point)
    FatJet_record = ak.zip(fatjet_obs, depth_limit=2)

    # branches to write: the FatJet record, plus one separate branch per Lund observable
    branches = {"FatJet": FatJet_record}
    for name, arr in lund_obs.items():
        branches[f"FatJet_{name}"] = arr

    with uproot.recreate(friend_fname) as fout:
        fout.mktree("DelphesFriend", {name: ak.type(arr) for name, arr in branches.items()})
        fout["DelphesFriend"].extend(branches)

    if verbose:
        print(f"Wrote friend tree with {n_evt} events and {n_jets} Lund jets: {friend_fname}")