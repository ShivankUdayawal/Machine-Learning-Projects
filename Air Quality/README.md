# Air Quality 

![](https://github.com/ShivankUdayawal/Machine-Learning-Projectcs/blob/main/Air%20Quality/image.jpg)

### Introduction
Recent economic and social developments have had aneffect on various environmental variables, including the land,water resources, and air. Because of this, wireless sensornetwork-based air quality monitoring is a popular researchtopic. According to WHO, seven million deathswere related to air pollution each year. Based on the computation of pollutants that are harmfulto human health, the air pollution level index can be created. The Air Quality Index (AQI) is the name of this index, which ranges from 0 to 500. A high AQI is not good for prople. There are distinct ways to calculate the AQI, such asusing the formula or using machine learning techniques. In 2018, study led by Samir Lemes and colleagues demonstratedthe disparity between several approaches to estimate the AQI by calculating and ranking AQI values according to certaincriteria. They then used these parameters to calculate thelevels of air pollution in two different parts of Bosnia andHerzegovina. The ﬁnal result of their work illustrates thecomparison of AQI values on the same dataset, which wasobtained by using different methods of US AQI, EU AQI, andSAQI 11 standards.

### Data
The dataset contains 9358 instances of hourly averaged responses from an array of 5 metal oxide chemical sensors embedded in an Air Quality Chemical Multisensor Device. The device was located on the field in a significantly polluted area, at road level,within an Italian city. Data were recorded from March 2004 to February 2005 (one year)representing the longest freely available recordings of on field deployed air quality chemical sensor devices responses. Ground Truth hourly averaged concentrations for CO, Non Metanic Hydrocarbons, Benzene, Total Nitrogen Oxides (NOx) and Nitrogen Dioxide (NO2)  and were provided by a co-located reference certified analyzer. Evidences of cross-sensitivities as well as both concept and sensor drifts are present as described in De Vito et al., Sens. And Act. B, Vol. 129,2,2008 (citation required) eventually affecting sensors concentration estimation capabilities. Missing values are tagged with -200 value.
This dataset can be used exclusively for research purposes. Commercial purposes are fully excluded.

### Attribute Information
1.  Date (DD/MM/YYYY)
2.  Time (HH.MM.SS)
3.  True hourly averaged concentration CO in mg/m^3 (reference analyzer)
4.  PT08.S1 (tin oxide) hourly averaged sensor response (nominally CO targeted)
5.  True hourly averaged overall Non Metanic HydroCarbons concentration in microg/m^3 (reference analyzer)
6.  True hourly averaged Benzene concentration in microg/m^3 (reference analyzer)
7.  PT08.S2 (titania) hourly averaged sensor response (nominally NMHC targeted)
8.  True hourly averaged NOx concentration in ppb (reference analyzer)
9.  PT08.S3 (tungsten oxide) hourly averaged sensor response (nominally NOx targeted)
10. True hourly averaged NO2 concentration in microg/m^3 (reference analyzer)
11. PT08.S4 (tungsten oxide) hourly averaged sensor response (nominally NO2 targeted)
12. PT08.S5 (indium oxide) hourly averaged sensor response (nominally O3 targeted)
13. Temperature in Â°C
14. Relative Humidity (%)
15. AH Absolute Humidity


