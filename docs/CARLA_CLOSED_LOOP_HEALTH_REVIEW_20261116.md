# Read-only closed-loop health review
Sources: artifacts/carla-miniloop-closed-{old,new}-health report/records/frames/live, data/carla_miniloop_cache_v1 training428windows. No additional inference, training, simulation, controller or guard changes.

## Evidence
Bothfirstpolicycallsstartat0m/s afterbraked3.6ssensorwarmup. Thetrainingcollectoracceleratedimmediatelyafterspawn; adapterexcludedfirst3s duefalling/invalidnavigationtransients. Among428newtrainingwindows,61initialspeed<.5m/s; all61end5sspeedexact0, futureforwardrange0..0.0105m. Numberv0<.5andfutureend>2m/s=0. Thus modelnevertrainedsettledstandstill->go inthisdataset. Doesnotproveuniquecause; speed-onlystatewithheldcommandsalsomakesstop/gointentambiguous.

Newfirstcandidateactualpoints: .2s(x=.00986,y=-.000084,v=.0986),.6s(.06263,.01377,.1189),5s(-.58239,.15528,.40465). Guardusesnearest4mlookaheadwhichbecomesendpoint; x<-.5 triggers. Backwardendpointrealprediction,notjusttextualguarderror. Candidatewasnotexecuted; cannotblamevehicletrackingfornewfailure. Do notrelaxthresholdorselectafavorablecandidatewithoutpredeclaredrule.

Old50executedplanningcalls over5s: desiredspeedfrom.6sprediction.792–1.371m/s,throttle.1409–.1982,firststeer-.5saturated. Firstcandidate5sendpoint(4.565,2.419),currenttargetspeed.792. Maxactual.808m/s and2.224mprogress areconsistentwithlowmodeltargetspluscontroller response; notpurecontrollerfailure. Itdidnotreachnormalorbrakingphase. Steering/dynamicsarealsoimperfect.

Onlinefirsthistoryknownvoxelcoverage.007202(0.7202%)forbothold/new. Trainmean.007673,range.007454–.008071. Verysparsecausalrayhits, notonlineall-empty. Exacthistoryage[.6,.4,.2,0],sameadapterandmaprange,syntheticaxespass. Thesechecksdo notcertifyscene semantics/oraclelabelsorfieldgeometry; sourceego-boxfilterandsemanticGTlimitsremain.

Workerusesfixedseed2301,K1,Euler16. Onecandidateperstate:noevidenceofrandom-bad-luckfrequency. Existingrecordreviewcannotseparatenoisefailurefromsystematicbiaswithoutfrozenmulti-noisereplay; notperformedinthisreview.

## Minimal next action proposal, not executed
First collect a tiny matched settled-start segment: brake and settle before recording, then normalforwardacceleration withroutegoal, plus existingmoving/decelerationsegments. Separatetrain/dev episodes beforewindows; allowcomplete.6shistory and5slabels. Preserveguards. Verifyatleastpositivecountofsettledv<.5->movingtargetsandstateintentcontract. IfsameOCC/speedobservationscanmeanstaystoppedorgo, dataaloneisambiguous; addingexplicitroute/desiredbehaviorrequiresseparatecontrolledchange,notimplicitlabelleakage.

Thenboundedpairedsamebudgetdata-onlyablation with/withoutstartcoverage, retainingstopsandnormalprogress, evaluatefixednoiseandpredeclaredmultinoisesofflinebeforeclosedloop. Nochangingguardorwarminitialspeedtosilentlybypassfailure. Arollinghandofftestmayseparatelyinspectin-distributionnormaldriving,butcannotcountasstartpass. Noevidencejustifiesmorelayers,longtraining,orclaimingnewdatafixedFlow.
