# SC-Elephant (Single-Cell Extremely Large Data Analysis Platform)
![scelephant-logo](doc/img/scelephant_logo.png)



`SC-Elephant` utilizes `RamData`, a novel single-cell data storage format, to support a wide range of single-cell bioinformatics applications in a highly scalable manner, while providing a convenient interface to export any subset of the single-cell data in `SCANPY`'s `AnnData` format, enabling efficient downstream analysis the cells of interest. The analysis result can then be made available to other researchers by updating the original `RamData`, which can be stored in cloud storage like `AWS` (or any AWS-like object storage).



`SC-Elephant` and `RamData` enable real-time sharing of extremely large single-cell data using a browser-based analysis platform as it is being modified on the cloud by multiple other researchers, convenient integration of a local single-cell dataset with multiple large remote datasets (`RamData` objects uploaded by other researchers), and remote (private) collaboration on an extremely large-scale single-cell genomics dataset. 



A <tt>RamData</tt> object is composed of two <b><tt>RamDataAxis</tt></b> (<b>Axis</b>) objects and multiple <b><tt>RamDataLayer</tt></b> (<b>Layer</b>) objects.



![scelephant-logo](doc/img/scelephant.js.structure.png)



The two RamDataAxis objects, <b>'Barcode'</b> and <b>'Feature'</b> objects, use <b><tt>'filter'</tt></b> to select cells (barcodes) and genes (features) before retrieving data from the <tt>RamData</tt> object, respectively.



For a demonstration of the use of `RamData` object on a web browser, please visit http://scelephant.org/


