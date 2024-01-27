#adapted from: https://xaodanahelpers.readthedocs.io/en/latest/UsingUs.html

from xAODAnaHelpers import Config
c = Config()

c.algorithm("BasicEventSelection", {"m_truthLevelOnly": False,
#                                    "m_applyGRLCut": True,
#                                    "m_GRLxml": "$ROOTCOREBIN/data/xAODAnaHelpers/data12_8TeV.periodAllYear_DetStatus-v61-pro14-02_DQDefects-00-01-00_PHYS_StandardGRL_All_Good.xml",
                                    "m_doPUreweighting": False,
                                    "m_vertexContainerName": "PrimaryVertices",
                                    "m_PVNTrack": 2,
                                    "m_useMetaData": False,
                                    "m_triggerSelection": "L1_.*|HLT_.*",
                                    "m_storeTrigDecisions": True,
                                    "m_storePassL1": True,
                                    "m_storePassHLT": True,
                                    "m_name": "myBaseEventSel"})

c.algorithm("TreeAlgo", {
                         "m_debug": True,
                         "m_name": "EB_Tree",
                         "m_jetContainerName": "HLT_AntiKt4EMPFlowJets_subresjesgscIS_ftf",
                         "m_jetBranchName": "HLT_jet",
                         "m_jetDetailStr": "kinematic",
                         "m_photonContainerName": "HLT_egamma_Photons",
                         "m_photonDetailStr": "kinematic",
                         "m_elContainerName": "HLT_egamma_Electrons HLT_egamma_Electrons_LRT",
                         "m_elBranchName": "HLT_el HLT_el_LRT",
                         "m_elDetailStr": "kinematic",
                         "m_muContainerName": "HLT_MuonsIso HLT_MuonsCB_LRT",
                         "m_muBranchName": "HLT_muon HLT_muon_LRT",
                         "m_muDetailStr": "kinematic",
			 "m_l1JetContainerName": "L1_jFexSRJetRoI_OfflineCopy L1_jFexLRJetRoI_OfflineCopy L1_gFexSRJetRoI_OfflineCopy L1_gFexLRJetRoI_OfflineCopy LVL1JetRoIs",
                         "m_l1JetBranchName": "L1_jFexSRJet L1_jFexLRJet L1_gFexSRJet L1_gFexLRJet L1_Jets",
			 "m_TrigMETContainerName": "HLT_MET_pfopufit",
                         "m_TrigMETDetailStr": "kinematic",
                         "m_trigDetailStr": "basic passTriggers"
                         })


