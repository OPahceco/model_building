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


def ecf(cs, npoint, angles, beta):
    # generalized ECF of each jet when it is clustered exclusively into one jet: one value per jet
    values = cs.exclusive_jets_energy_correlator(
        njets=1, beta=beta, npoint=npoint, angles=angles, func="generalized"
    )
    values = ak.fill_none(values, np.nan)
    return ak.to_numpy(ak.flatten(values, axis=None)).astype(np.float64)


def ratio(numerator, denominator, fill_value):
    # numerator / denominator**2, or fill_value where it cannot be computed
    with np.errstate(divide="ignore", invalid="ignore"):
        result = numerator / denominator**2
    good = (
        np.isfinite(numerator)
        & np.isfinite(denominator)
        & (denominator > 0.0)
        & np.isfinite(result)
    )
    return np.where(good, result, fill_value)


def calculate_n2_n3_fastjet(FJ, beta=1.0, fill_value=SENTINEL):
    """
    Generalized ECF observables of each FatJet:

        N2 = (2e3) / (1e2)^2    (needs at least 3 constituents)
        N3 = (2e4) / (1e3)^2    (needs at least 4 constituents)

    Jets that cannot be assigned a value get fill_value.
    Output shape: event -> FatJet
    """
    jets = ak.flatten(FJ, axis=1)  # all jets of all events in one list
    n_const = count_constituents(jets)

    n2 = np.full(len(jets), fill_value, dtype=np.float64)
    n3 = np.full(len(jets), fill_value, dtype=np.float64)

    # The constituents already belong to one FatJet. The large radius ensures that
    # FastJet combines all of them into one exclusive jet.
    sel = n_const >= 3
    if np.any(sel):
        cs = cluster_jets(jets[sel], R=1000.0)
        n2[sel] = ratio(ecf(cs, 3, 2, beta), ecf(cs, 2, 1, beta), fill_value)

    sel = n_const >= 4
    if np.any(sel):
        cs = cluster_jets(jets[sel], R=1000.0)
        n3[sel] = ratio(ecf(cs, 4, 2, beta), ecf(cs, 3, 1, beta), fill_value)

    # back to event -> FatJet
    n_per_event = ak.to_numpy(ak.fill_none(ak.num(FJ, axis=1), 0))
    N2 = ak.unflatten(n2.astype(np.float32), n_per_event)
    N3 = ak.unflatten(n3.astype(np.float32), n_per_event)
    return ak.to_packed(N2), ak.to_packed(N3)


def calculate_primary_lund_xy_fastjet(FJ, R=0.8, min_jet_pt=20.0):
    """
    Primary Lund-plane points of each FatJet, using FastJet.
    Jets need pT >= min_jet_pt and at least 2 constituents.

    Output shape: event -> Lund point (all jets of an event are put together)
    """
    jets = ak.flatten(FJ, axis=1)  # all jets of all events in one list
    n_const = count_constituents(jets)
    pt = ak.to_numpy(ak.fill_none(jets["PT"], 0.0))
    used = (pt >= min_jet_pt) & (n_const >= 2)
    n_jets = int(np.count_nonzero(used))

    n_points = np.zeros(len(jets), dtype=np.int64)  # number of Lund points of each jet
    x_flat = np.zeros(0, dtype=np.float64)
    y_flat = np.zeros(0, dtype=np.float64)

    if n_jets > 0:
        cs = cluster_jets(jets[used], R=R)

        # Recluster the constituents into one exclusive C/A jet,
        # then obtain its primary Lund declustering sequence.
        lund = cs.exclusive_jets_lund_declusterings(njets=1)

        if "Delta" not in lund.fields or "kt" not in lund.fields:
            raise RuntimeError(
                "FastJet Lund output does not contain the expected "
                f"'Delta' and 'kt' fields. Found: {lund.fields}"
            )

        delta = lund["Delta"]
        kt = lund["kt"]

        # make sure there is exactly one list of declusterings per jet
        while delta.ndim > 2:
            delta = ak.flatten(delta, axis=1)
            kt = ak.flatten(kt, axis=1)

        good = np.isfinite(delta) & np.isfinite(kt) & (delta > 0.0) & (kt > 0.0)
        x = np.log(R / delta[good])
        y = np.log(kt[good])

        n_points[used] = ak.to_numpy(ak.num(x, axis=1))
        x_flat = ak.to_numpy(ak.flatten(x, axis=1))
        y_flat = ak.to_numpy(ak.flatten(y, axis=1))

    # jet -> point, then event -> jet -> point, then all the points of an event together
    n_per_event = ak.to_numpy(ak.fill_none(ak.num(FJ, axis=1), 0))
    LundX = ak.unflatten(ak.unflatten(x_flat, n_points), n_per_event)
    LundY = ak.unflatten(ak.unflatten(y_flat, n_points), n_per_event)
    LundX = to_f32_jagged(ak.flatten(LundX, axis=2))
    LundY = to_f32_jagged(ak.flatten(LundY, axis=2))

    return LundX, LundY, n_jets

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

    # N2, N3
    N2, N3 = calculate_n2_n3_fastjet(FJ, beta=1.0, fill_value=SENTINEL)
    jets_like = {"N2": N2, "N3": N3}

    # primary Lund plane points. Shape: event -> Lund point
    FatJet_LundX, FatJet_LundY, n_jets = calculate_primary_lund_xy_fastjet(
        FJ, R=0.8, min_jet_pt=20.0
    )

    # make sure the friend tree has the same number of entries as the Delphes tree
    all_arrays = dict(jets_like, LundX=FatJet_LundX, LundY=FatJet_LundY)
    for name, arr in all_arrays.items():
        if len(arr) != n_evt:
            raise RuntimeError(
                f"{name} has {len(arr)} events but Delphes has {n_evt}: "
                "do not drop or reorder events when computing friend branches."
            )

    # force all jet-level observables to use the exact FatJet layout
    jets_like = {name: force_fatjet_layout(arr, FJ) for name, arr in jets_like.items()}

    # Zip the jet-level variables into one FatJet record, so there is only one counter
    # per group structure (event -> jet, and event -> Lund point)
    FatJet_record = ak.zip(
        {"N2": jets_like["N2"], "N3": jets_like["N3"]},
        depth_limit=2,
    )

    with uproot.recreate(friend_fname) as fout:
        fout.mktree(
            "DelphesFriend",
            {
                "FatJet": ak.type(FatJet_record),
                "FatJet_LundX": ak.type(FatJet_LundX),
                "FatJet_LundY": ak.type(FatJet_LundY),
            },
        )

        fout["DelphesFriend"].extend(
            {
                "FatJet": FatJet_record,
                "FatJet_LundX": FatJet_LundX,
                "FatJet_LundY": FatJet_LundY,
            }
        )

    if verbose:
        print(f"Wrote friend tree with {n_evt} events and {n_jets} Lund jets: {friend_fname}")
