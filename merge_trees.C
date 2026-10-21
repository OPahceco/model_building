#include "TFile.h"
#include "TTree.h"
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

  // Readers
  TTreeReader rMain(tm);
  TTreeReader rFr(tf);

  // Build a reader for each requested branch; all of them must exist
  struct FriendItem {
    std::string inName;                 // e.g. "FatJet_N2"
    std::string outName;                // e.g. "F_FatJet_N2"
    std::unique_ptr<TTreeReaderArray<Float_t>> rdr;  // leaf-counted array
    std::vector<float> buffer;          // output buffer
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

  // --- Output: clone full Delphes structure, connect addresses to tm ---
  std::unique_ptr<TFile> fout(TFile::Open(fOut, "RECREATE"));
  if (!fout || fout->IsZombie()) {
    std::cerr << "Cannot create " << fOut << "\n";
    return fail(3);
  }
  TTree* tout = tm->CloneTree(0);   // clone structure only
  tout->SetName(tMain);             // keep familiar name
  // Make tout's branches read from tm's memory so Fill() writes full Delphes
  tout->CopyAddresses(tm);

  // Add new friend branches to tout (pointing at our std::vector buffers)
  for (auto& it : items) {
    tout->Branch(it.outName.c_str(), &it.buffer);
  }

  // --- Event loop: sync main and friend, copy arrays, and fill once per event
  const Long64_t n = tm->GetEntries();
  for (Long64_t i = 0; i < n; ++i) {
    // advance both readers/trees in lockstep
    if (!rMain.Next() or !rFr.Next()) break;

    // copy each friend array into the vector buffer
    for (auto& it : items) {
      auto& rdr = *it.rdr;
      it.buffer.assign(rdr.begin(), rdr.end());
    }

    // Also load the main entry so tm's branch buffers are set
    tm->GetEntry(i);

    // write full Delphes + friend vectors in one go
    tout->Fill();
  }


  fout->cd();
  tout->Write(tMain, TObject::kOverwrite);
  fout->Close();



  if (verbose) {
    std::cout << "Wrote " << fOut << " with full Delphes + "
              << items.size() << " friend branches (prefixed F_*)\n";
  }
  return 0;
}