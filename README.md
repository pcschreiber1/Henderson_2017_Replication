# OSE data science final project | Philipp Schreiber | Summersemester 2021

**Henderson, J.V., Storeygard, A., Deichmann, U. (2017)** [Has climate change driven urbanization in Africa?](https://doi.org/10.1016/j.jdeveco.2016.09.001) *Journal of Development Economics* 124, 60–82.

Henderson et al. investigate the impact of increasing aridity on urbanization in Sub-Saharan Africa. In particular, the authors address the questions of i) whether adverse changes in climate induce a push from rural to urban areas and ii) how this within-country migration affects incomes in cities. They attribute previous studies' mixed findings to aggregation effects resulting from using national level data and, therefore, use variation at the district and city level. To enable such an analysis the authors constructed a panel data set combining district level census data, fine resolution climate data and night time lights (NTLs) between 1960 and 2008. The authors' central finding is that climate shocks only produce rural-urban movements if cities are can offer alternative employment options. The extension seeks to contribute to the development of a spatially explicit counterfactual framework: a spatial perspective on SUTVA is developed, tested and then applied to the analysis of Henderson et. al (2017).

The replication is carried out on the main notebook `Replication_notebook` in this repository. The auxiliary folder contains different functions made for loading and processing the spatial data, tables and plots. The replication can also be visualized using nbviewer and mybinder:

To ensure the reproducibility of the project, the repository is supported by a GitHub Actions Continuos Integration (CI) workflow. Its current state  can be found here:

</a>
<a href="https://github.com/OpenSourceEconomics/ose-data-science-course-project-pcschreiber1/actions/workflows/ci.yml"
    target="_parent">
    <img align="center"
       src="https://github.com/OpenSourceEconomics/ose-data-science-course-project-pcschreiber1/actions/workflows/ci.yml/badge.svg"
       width="200" height="20">
</a>

# Key References

* **Kolak, M., Anselin, L., 2019**. [A Spatial Perspective on the Econometrics of Program Evaluation](https://doi.org/10.1177/0160017619869781). *International Regional Science Review* 43, 128–153.

* **Rey, S., Arribas-Bel, D., Wolf, L., 2020**. [Geographic Data Science with Python](https://geographicdata.science/book/intro.html), *Jupyter Book*. ed.
