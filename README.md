# Detecting trees within the Tree Preservation Order using Multi-Input Deep Learning
Tree Preservation Order (TPO) is used to protect specific trees from damage and destruction. However, the current criteria and process of determining a TPO are vague and subjective. TPO data in many districts is stored in paper form, with a cumbersome and lengthy application process. It is creating a barrier to both urban development and environmental protection. This research collected and analysed TPO data status in Greater London and developed a multi-input deep learning model to detect potential TPOs in areas without public TPO data. The new method is more efficient and accurate than traditional methods. This research is helpful for governments, real estate developers and environmental organisations and can be extended to other areas and detect other social attributes of trees.

All the codes used in this project have been included in this repository. There are 10 jupyter notebook files in total; 8 of them are for model training and model evaluating, the results can also be found in the folder namely tpo_html_files, the other 2 of jupyter botebook files are for data preprocessing, including geographical data namely data_preprocess.ipynb and imagery data namely image_data.ipynb. In the tpo_module folder, two python files have been shared. One of them is the function used in this project to address the dataset namely datasets.py, another includes functions which are used to create different machine learning models. At last, all the trained models have been stored in the folder called tpo_trained_model.

Besides, tables below show the detailed information of all data used in this research. 

TPO location	
	
	Barnet	02/2021	OPEN Barnet - Tree Preservation Orders	https://open.barnet.gov.uk/dataset/e5nge/tree-preservation-orders-schedule-items
	
	Lambeth	02/2021	Lambeth Open Mapping Data - london borough of lambeth	https://lambethopenmappingdata-lambethcouncil.opendata.arcgis.com/datasets/tree-preservation-order-points?geometry=-0.290%2C51.423%2C0.056%2C51.497
	
	Hammersmith and Fulham	05/2019	Shared GIS and Location Services 	https://data-lbhf.opendata.arcgis.com/datasets/c15c9a3aca054d4cb7c6b3f84779d64d_1
	
	Richmond upon Thames	06/2017	 Richmond upon Thames Borough Council	https://www.whatdotheyknow.com/request/tree_preservation_orders_digital_5
	
	Redbridge	2017	 Redbridge Borough Council	https://data.redbridge.gov.uk/View/planning-and-land/tpo-addresses#

The Greater London Tree canopy 	
	
	Greater London	10/2020	Geomni - UKMap(London Only) - UKMap*	https://digimap.edina.ac.uk/roam/download/geomni

Aerial Imagery (25cm resolution)	
	
	Barnet	2016	Digimap - Aerial - Aerial Imagery(Latest) - High Resolution(25cm)*	https://digimap.edina.ac.uk/roam/download/aerial
	
	Lambeth	2015(part) 2016(part)		
	
	Hammersmith and Fulham	2015		
	
	Richmond upon Thames	2015		
	
	Redbridge	2018		

Geographic data, Greater London	
	
	Road network		03/2019	OS MaterMap - Highways -All (FGDB Network)*	https://digimap.edina.ac.uk/roam/download/os
	
	Green Space		10/2020	OS MasterMap - Greenspace*	
	
	river centerline	02/2021	OS MaterMap - Water network*	
	
	Buildings		03/2018	OS MaterMap - Building Height Attribute*	
	
	list buildings		03/2022	data.gov.uk - Listed Buildings GIS Data	https://data.gov.uk/dataset/8db67112-67b0-43f2-b863-2ac9c58d52bf/listed-buildings-gis-data
	
	nature reserve		03/2021	data.gov.uk - Local Nature Reserves (England)	https://data.gov.uk/dataset/acdf4a9e-a115-41fb-bbe9-603c819aa7f7/local-nature-reserves-england
	
	Indices of Deprivation	2019	London Datastore - Indices of Deprivation	https://data.london.gov.uk/dataset/indices-of-deprivation

> *: Those kinds of data are not from open source, licenses are needed to use those datasets. 
