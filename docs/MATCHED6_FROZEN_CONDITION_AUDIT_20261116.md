# Matched-six frozen start/condition audit, 2026-11-16

No optimizer update, no new training, no CARLA execution. Existing checkpoint hashes unchanged. Sources: scripts/audit_matched6_start_condition.py and scripts/audit_matched6_frozen_response.py; artifacts/matched6-start-frozen-audit/{report,rows,frozen_response}.json.

## Correction to previous interpretation
The16devwindows described as currentleadalready moving (>0.2m/s) are ONLYthefirst0..0.3s after eachof4leadonsets. Past.6s truelead displacement is about.015,.056,.122,.213m, below.8m voxelresolution and comparable tohistorical centroidnoise. Objectvisibility isNOTreliablemotionobservability.

The actual4frameOCC no-IDreadout isvalid16/16, but only1/16 exceeds theOLDtrain-only fixed .698678m apparent-displacement threshold. This conservative detector isnotaninformation-theoretic bound: subvoxel occupancy changes mayexist. Alsoassociationvaliddoesnotcertifycorrecttarget. NoGTID/posewas usedbyreadout;GTusedonlyposthoc.

Train has32corresponding moving-lead/low-egospeedwindows across4episodes and8onsets;only7crossoldmotionthreshold. Actual512step exposure91/93moving-lead samples,19/20above-thresholdsamples acrossseeds7201/7202. Allfuture-start labels186train/91dev include many casesbeforeleadbeginsmoving. Futurelead schedule isunknown toactor: donotjudgeall5s missesasunresponsivepolicy.

Currentpermission istrue, currentcruise constant, historyagesactual[-.6,-.4,-.2,0]representedpositiveage[.6,.4,.2,0], currentandpreviouscommandvaluesagree. Nohiddenpermissiontransitionremains. Nav/sync/GTpoisonpassedearlier. Sourcecruise/routedistributionconfoundsremain.

## A distinct failure remains
For16devcurrentmoving-lead windows, label speeds at[.2,.6,1,5]s are[.213,.728,1.435,6.521]. Pred7201=[.599,.586,.577,.567],7202=[1.011,.965,.928,.948]. Thusnot literallyzero/noaction: theyjump tooquicklyinitiallybut failtoformanincreasingtrajectory. Five-secondendpointevaluationalonehidsthis.

81devwindows whosefirst1slabeledspeedis<.2 have labelmeans[0,0,.0067]at.2/.6/1s. Pred7201=[.557,.532,.515],7202=[.965,.920,.883]. Prematuremotionexistswhen immediatehold isunambiguous. Constant/stationarybaselinesarebetterforhold;allstaticbaselinefailsprogresselsewhere.

Deceleration910train/446dev,trainexposure2563eachseed: notmerelyfewstopexamples. Devlabel[.2,.6,1,5]speeds[4.856,4.592,4.311,1.465];pred7201[4.087,4.052,4.029,4.005],7202[4.728,4.676,4.644,4.658]. Nearconstantfuturevelocity ratherthantemporalresponse issharedacrossgroups.

## Frozen intervention evidence
12deterministicallyspacedactualexamplespergroup, bothtrainanddev; no labelenteredforward. ReplacefourhistoryOCCwithcurrentframe repeated (removesobservedmotion), setOCCunknown, orzeropast-egospeed-derivedacceleration. TheseareOODablationinputs, not deployableconditions or causalproof.

Dev early-start predictedwithintrajectoryvxstd .0516/.0453m/s versuslabelspeedstd2.132. Train early-start .0510/.0460 versus2.265. Devdeceleration .0526/.0459 versus1.347. Thisflatnessalreadyexistsontrain, soheldroutealonecannotexplainit.

RepeatingcurrentOCC changesmeanvectorvelocity .0127/.00849m/s ondev earlystart and.0718/.00881ondev observed-motiondeceleration. Zeroinghistoricalegoacceleration changes.402/.394and.965/.952respectively. Unknown-OCC alsochangesoutputs,butlessconsistently. Modelrespondsstronglytoego dynamics yetweaklytofourframeOCCmotion intheseprobes; doesnotproveOCCunusedoridentifywhichencoderlayerlosesit.

## Representation / supervision / budget
Existing25x2normalizedvelocitycanrepresentdifferentvelocityateachtime; speedencoding/decodingroundtrip passed. It doesnotforceconstantvelocity. Decoderincludeslearned25positionembeddings; deterministicxtandFMtimearezero, sofuturetimeidentitycomesfromthoseembeddings, notdiffusiontime. Noevidencehereofwronglabeltimeindexorpermissionoff. Learned outputs behavealmostlikeonevelocityrepeatedacrosstime.

DirectqMSE approximatesconditionalmeans; unknownfutureleaddecisionsandshortpredictablehistorycanencourageaveraging. Thisdoesnotexcuseprematurefirstsecondmotionorallknowncurrentdecelerationerrors. Noablationyetidentifieswhetherpositionconditioningoptimization,sharedworldobjectiveorinsufficienttrainingdominates.

512updates coveronly2.82trainpasses, no convergencecertification. Fit-sampledtrainerrors1.77/1.97m closedev1.72/2.10; thisisnotasimpletrain-good/dev-badoverfittingsignature. Trainingbudgetmaymatter,butmerelydoublingstepshasnotbeentestedandisnotjustifiedasafixbythisaudit.

## Minimal next recommendation (not executed)
Do NOT collectmorejustbecause16currentleadcaseslookbad,anddonotchangemultiplestructures. The next smallest falsifiablecheck is a frozen per-future-time action/position-loss andgradient audit: compare residualerrors across25times, positionembedding gradients versus sharedoutput/condition gradients, and learnedphysical-time variation. Samefixedactualtrainwindows, nooptimizerupdate. Determinewhethercurrentsupervisionproducesa usabletime-varyingsignal before proposingone single-variabletrainingrepair. Keepuncertainfutureleadonsetsseparatefromimmediatehold/currentdeceleration. Noautomatictrainorclosedloop.
