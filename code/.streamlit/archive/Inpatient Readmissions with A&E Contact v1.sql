WITH AnEContact AS (SELECT DISTINCT

	PT.NHSNumber	
	,ReferralReceivedDateTime AnEArrivalDateTime
     
FROM Informatics_SSAS_Live.dbo.FactReferrals RF

	INNER JOIN Informatics_SSAS_Live.dbo.DimPatient PT
	ON RF.PatientID = PT.PatientID

	INNER JOIN Informatics_SSAS_Live.dbo.DimTeam TM
	ON RF.ReferredtoStaffTeamID = TM.TeamID
	
WHERE TM.TeamID IN (10931,10932,10933))

SELECT * FROM

(SELECT DISTINCT

      --T1.Team AS AdmissionWard
	  --,T2.Team AS CurrentWard
	  --,T3.Team AS DischargeWard
	  --,SP.MetricReAdmissions
	  --,SP.MetricLengthofStay
	  PT.NHSNumber
      ,ReAdmission = SP.ReAdmissionsFlag
	  ,AnEContact = CASE WHEN DATEDIFF(DD,AE.AnEArrivalDateTime,SP.AdmissionDateTime) <= 30 THEN 1 ELSE 0 END
	  ,AE.AnEArrivalDateTime
	  ,SP.AdmissionDateTime
      ,CASE 
		   WHEN CONVERT(CHAR(7), ONS.[Index of Multiple Deprivation (IMD) Decile (where 1 is most deprived 10% of LSOAs)]) is null THEN 99				--deprevation based on patient postcode.  Added 18/1/21 by DW as requested by Phil Horner
		   ELSE CONVERT(INT, ONS.[Index of Multiple Deprivation (IMD) Decile (where 1 is most deprived 10% of LSOAs)])
	   END AS DeprivationIndex
	  ,PT.Sex AS Gender -- one hot code
	  ,PT.Age
	  ,PT.AgeGroup
	  ,MaritalStatus = REPLACE(REPLACE(PT.MaritalStatus,'[',''),']','')
	  --,PT.Religion
	  --,SexualOrientation -- one hot code -- data for this is very limited
	  ,PT.Ethinicity AS Ethnicity -- one hot code
	  ,LearningDisability = CASE WHEN PT.LearningDisability = 'Yes' THEN 1 ELSE 0 END
	  ,AutismDiagnosis = CASE WHEN PT.AutismDiagnosis = 'Yes' THEN 1 ELSE 0 END
	  ,ExBAF = CASE WHEN PT.BritishArmedForcesIndicatorId = '02' THEN 1 ELSE 0 END
	  ,AccommodationStatus = CASE WHEN AccommodationStatus IS NULL THEN 'Unknown' ELSE AccommodationStatus END
	  ,RANK() OVER (PARTITION BY PT.NHSNumber ORDER BY AE.AnEArrivalDateTime DESC) AS ArriveRank
	  
  FROM Informatics_SSAS_Live.dbo.FactInpatientSpells SP
	  
	  INNER JOIN Informatics_SSAS_Live.dbo.DimPatient PT
	  ON SP.PatientID = PT.PatientID

	  INNER JOIN Informatics_SSAS_Live.dbo.DimTeam T1
	  ON SP.AdmissionWardID = T1.TeamID

	  LEFT OUTER JOIN Informatics_SSAS_Live.dbo.FactInpatientEpisodes EP
	  ON SP.PatientID = EP.PatientID
	  AND SP.SourceInpatientSpellID = EP.InpatientSpellID
	  AND EP.EpisodeInSpellCount = 1

	  LEFT OUTER JOIN Informatics_SSAS_Live.dbo.FactReferrals RF
	  ON SP.ReferralID = RF.ReferralID
	  AND SP.PatientID = RF.PatientID

	  LEFT OUTER JOIN Informatics_SSAS_Live.dbo.DimReferralSource DRS			(NOLOCK)		
	  ON RF.ReferralSourceID = DRS.ReferralSourceID

	  LEFT JOIN GlobalLookupTables.dbo.v_ONS_GOV_DataMart ONS					(NOLOCK)		--Added by DW 18/1/21 as requested by PH off the back of a request by NHSE to monitor this type of data more closely
	  ON PT.Postcode = ONS.pcds

	  LEFT OUTER JOIN Informatics_SSAS_Staging.NDS.NationalCodeMasterLookup AS MB
	  ON PT.BritishArmedForcesIndicatorId = mb.LocalCode
	  AND MB.ElementName = 'ExBAFIndicator'

	  LEFT JOIN AnEContact AE
	  ON PT.NHSNumber = AE.NHSNumber
	  AND AE.AnEArrivalDateTime <= SP.AdmissionDateTime
	  AND DATEDIFF(DD,AE.AnEArrivalDateTime,SP.AdmissionDateTime) <=30

WHERE SP.AdmissionDate >= '20200101'
	  AND SP.SourceSystemID IN (3,9) 
	  AND SP.RecordArchived = 0
	  AND SP.PatientID > -999
	  AND T1.CostCentreCode IS NOT NULL
	  AND PT.NHSNumber <> '[novalue]'
	  AND (CASE WHEN T1.SourceSystemID = 3 AND  T1.SourceWardID IN 
				 (
					'337', -- Ramsey Unit duplicate    (PH 11/5/21)
					'342' -- Kentmere Ward - Detox
			
				) THEN 'N'

				WHEN T1.SourceSystemID = 9 AND  T1.SourceWardID IN 
				(
					 '419' -- zz_Bleasdale
				--	 '483', -- Section 136 Suite Dane Garth
				--	 '484' -- Dova Ward
				)	 THEN 'N'
			
			 ELSE 'Y' 
			 END) = 'Y'

		) AE

		WHERE AE.ArriveRank = 1 -- Only most recent A&E visit to admission

	--ORDER BY ReAdmission DESC