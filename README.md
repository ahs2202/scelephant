![scelephant-logo](https://raw.githubusercontent.com/ahs2202/scelephant/master/doc/img/scelephant_logo.png)

# SC-Elephant (Single-Cell Extremely Large Data Analysis Platform)

[![PyPI version](https://badge.fury.io/py/scelephant.svg)](https://badge.fury.io/py/scelephant)

`SC-Elephant` utilizes `RamData`, a novel single-cell data storage format, to support a wide range of single-cell bioinformatics applications in a highly scalable manner, while providing a convenient interface to export any subset of the single-cell data in `SCANPY`'s `AnnData` format, enabling efficient downstream analysis of the cells of interest. The analysis result can then be made available to other researchers by updating the original `RamData`, which can be stored in cloud storage like `AWS` (or any AWS-like object storage).



`SC-Elephant` and `RamData` enable real-time sharing of extremely large single-cell data using a browser-based analysis platform as it is being modified on the cloud by multiple other researchers, convenient integration of a local single-cell dataset with multiple large remote datasets (`RamData` objects uploaded by other researchers), and remote (private) collaboration on an extremely large-scale single-cell genomics dataset. 



Tutorials can be found at `doc/jn/`

**[Tutorial 1) Processing and analysis of the 3k PBMCs dataset using SC-Elephant](https://scelephant-free.s3.amazonaws.com/doc/SC-Elephant_PBMC3k_processing_and_analysis_tutorials.html)**

**[Tutorial 2) Alignment of PBMC3k to the ELDB (320,000 cells subset) and cell type prediction using SC-Elephant](https://scelephant-free.s3.amazonaws.com/doc/SC-Elephant_PBMC3k_alignment_to_the_ELDB_subset_320k_tutorials.html)**

**[Tutorial 3) Combine 10x MEX count matrices memory-efficiently using SC-Elephant](https://scelephant-free.s3.amazonaws.com/doc/SC-Elephant_Combine_10x_MEX_Count_Matrices.html)**

**[Tutorial 4) Convert existing AnnData into RamData for collaborative data sharing](https://scelephant-free.s3.amazonaws.com/doc/SC-Elephant_Convert_AnnData_to_RamData_for_collaborative_data_sharing.html)**



Briefly, a <tt>RamData</tt> object is composed of two <b><tt>RamDataAxis</tt></b> (<b>Axis</b>) objects and multiple <b><tt>RamDataLayer</tt></b> (<b>Layer</b>) objects.

![ramdata_struc](https://raw.githubusercontent.com/ahs2202/scelephant/master/doc/img/ramdata_struc.png)



The two RamDataAxis objects, <b>'Barcode'</b> and <b>'Feature'</b> objects, use <b><tt>'filter'</tt></b> to select cells (barcodes) and genes (features) before retrieving data from the <tt>RamData</tt> object, respectively.

![ramdata_struc](https://raw.githubusercontent.com/ahs2202/scelephant/master/doc/img/ramtx_sparse_matrix.png)

`RamData` employs `RAMtx` (Random-accessible matrix) objects to store count matrix in sparse or dense formats.



`RamData` greatly simplify sharing of very large single-cell datasets on the Web. Once processed by SC-Elephant, `RamData` can be uploaded to GitHub, Amazon S3 Cloud, or any static file servers to share your single-cell datasets publicly with the research community or privately with your collaborators. The machine learning models, kNN graphs, cell-type annotations, and random-accessible expression count matrices (to name a few) of your single-cell datasets on the Web can be easily explored in Python environments and web browsers using SC-Elephant and **SC-Elephant.js**, respectively. 



To explore `RamData` objects publicly available on the Web using a web browser, please visit our [SC-Elephant DB Viewer](http://scelephant.org/).

![scelephant-js-example](https://raw.githubusercontent.com/ahs2202/scelephant/master/doc/img/scelephant_js_example.png)
