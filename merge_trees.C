#include "TFile.h"
#include "TTree.h"
#include "TBranch.h"
#include "TTreeReader.h"
#include "TTreeReaderArray.h"
#include "TSystem.h"
#include <vector>
#include <string>
#include <memory>
#include <iostream>

// Report failure to the caller: a plain "return" from a macro does not change the exit
// status of `root -q`, so exit explicitly to let run_model detect the error and stop
static int fail(int code) {
  gSystem->Exit(code);
  return code;
}

int merge_trees(
  const char* fMain    = "events.root",
  const char* fFriend  = "events_friend.root",
  const char* fOut     = "merged_tree.root",
  const std::vector<std::string>& branches = {},
  bool        verbose  = false,
  const char* tMain    = "Delphes",
  const char* tFr      = "DelphesFriend"
) {

  // Open inputs
  // TFile::Open() is used instead of the TFile constructor because the constructor
  // does not work properly with files accessed over xrootd
  std::unique_ptr<TFile> fm(TFile::Open(fMain));
  if (!fm || fm->IsZombie()) {
    std::cerr << "Cannot open " << fMain << "\n";
    return fail(1);
  }

  TTree* tm = (TTree*)fm->Get(tMain);

  std::unique_ptr<TFile> ff(TFile::Open(fFriend));
  if (!ff || ff->IsZombie()) {
    std::cerr << "Cannot open " << fFriend << "\n";
    return fail(1);
  }

  TTree* tf = (TTree*)ff->Get(tFr);

  if (!tm) {
    std::cerr << "[debug] Main tree not found: " << tMain << "\n";
    std::cerr << "[debug] Keys in main file:\n";
    fm->ls();
  }

  if (!tf) {
    std::cerr << "[debug] Friend tree not found: " << tFr << "\n";
    std::cerr << "[debug] Keys in friend file:\n";
    ff->ls();
  }

  if (!tm || !tf) {
    std::cerr << "Missing tree(s).\n";
    return fail(2);
  }

  // --- Which friend branches to add: list given by the caller ---
  if (branches.empty()) {
    std::cerr << "No friend branches requested.\n";
    return fail(5);
  }

  // The friend tree must have exactly one entry per Delphes entry
  const Long64_t n = tm->GetEntries();
  if (tf->GetEntries() != n) {
    std::cerr << "Entry mismatch: " << fMain << " has " << n << " entries, "
              << fFriend << " has " << tf->GetEntries() << "\n";
    return fail(4);
  }

  // Reader for the friend tree
  TTreeReader rFr(tf);

  // Build a reader for each requested branch; all of them must exist
  struct FriendItem {
    std::string inName;                 // e.g. "FatJet_N2"
    std::string outName;                // e.g. "F_FatJet_N2"
    std::unique_ptr<TTreeReaderArray<Float_t>> rdr;  // leaf-counted array
    std::vector<float> buffer;          // output buffer
    TBranch* branch = nullptr;          // new branch in the output tree
  };
  std::vector<FriendItem> items;
  items.reserve(branches.size());
  for (const auto& nm : branches) {
    if (!tf->GetBranch(nm.c_str())) {
      std::cerr << "Requested friend branch not found: " << nm << "\n";
      return fail(6);
    }
    FriendItem it;
    it.inName  = nm;
    it.outName = "F_" + nm;     // prefix to avoid name collisions
    it.rdr     = std::make_unique<TTreeReaderArray<Float_t>>(rFr, nm.c_str());
    items.emplace_back(std::move(it));
  }

  // --- Output: copy the Delphes file as it is, then append the new branches to the copy ---
  // Cloning the tree entry by entry would rewrite the objects, which changes ROOT's internal
  // reference IDs (fUniqueID) and breaks the references from jets to their constituents
  if (!TFile::Cp(fMain, fOut, kFALSE)) {
    std::cerr << "Cannot copy " << fMain << " to " << fOut << "\n";
    return fail(3);
  }
  std::unique_ptr<TFile> fout(TFile::Open(fOut, "UPDATE"));
  if (!fout || fout->IsZombie()) {
    std::cerr << "Cannot open " << fOut << " for update\n";
    return fail(3);
  }
  TTree* tout = (TTree*)fout->Get(tMain);
  if (!tout) {
    std::cerr << "Main tree not found in " << fOut << "\n";
    return fail(2);
  }

  // Add new friend branches to the tree (pointing at our std::vector buffers)
  for (auto& it : items) {
    it.branch = tout->Branch(it.outName.c_str(), &it.buffer);
  }

  // --- Fill only the new branches, one value per existing entry ---
  for (Long64_t i = 0; i < n; ++i) {
    if (!rFr.Next()) break;
    for (auto& it : items) {
      it.buffer.assign(it.rdr->begin(), it.rdr->end());
      it.branch->Fill();
    }
  }

  tout->Write("", TObject::kOverwrite);
  fout->Close();

  if (verbose) {
    std::cout << "Wrote " << fOut << " with full Delphes + "
              << items.size() << " friend branches (prefixed F_*)\n";
  }
  return 0;
}
