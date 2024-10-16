# Python module to handle the SDM2 version of the (S)ingle-(M)olecule (D)ataset (SMD) file format
We say that data from a single-molecule experiment is stored in an `SMD`. These exist in files (HDF5 format) or in memory as an object. This format is slighty different than originally described in: Greenfeld, M., van de Meent, JW., Pavlichin, D.S. et al. Single-molecule dataset (SMD): a generalized storage format for raw and processed single-molecule data. BMC Bioinformatics 16, 3 (2015). https://doi.org/10.1186/s12859-014-0429-4.

## Notable differences
* HDF5 file based instead of JSON
* Multiple SMDs can live in the same HDF5 file; each lives as a group within the HDF5 file
* Flexible WRT analysis: results from any analysis should be saved as a sub-group within the particular SMD.
* When molecules from different data sources can be combined into the same SMD, the original source is remembered.
* Classifying/grouping molecules is very important for many methods; each molecule in the SMD has an integer to specify the class to which it belongs.

## Instructions
Install in place using
```bash
cd <path to pysmd-hdf5 folder>
pip install -e './'
```

## Format Specifications/Description
Notes/Non-standard points are put in angle brackets.

### SMD (v2 in a Python Object)
Assume you have an SMD container object called `dataset`

```
dataset.raw : ndarray(NxTxD...) - the data
<!-- dataset.classes : ndarray(N) - integers of class to which each molecule is assigned -->
<!-- dataset.data_index : ndarray(N) - integers specifying current order of molecules in the SMD -->
dataset.source_index : ndarray(N) - integers specifying the source each molecule came from
dataset.source_names : list - the names of each sources
dataset.source_dicts : list - dictionaries with metadata entries for each source
dataset.smd_dict : dictionary - metadata entries describing smd file

dataset.nmol : int - the value of dataset.raw.shape[0]
dataset.nt : int - the value of dataset.raw.shape[1]
dataset.ntime : int - the value of dataset.raw.shape[1]
dataset.ncolor : int - the value of dataset.raw.shape[2]
dataset.ncolors : int - the value of dataset.raw.shape[2]
dataset.ndata - : int - the value of dataset.raw.shape[2:]
```

### SMD (v2) in an HDF5 File
Assume you have an HDF5 file with handle `f` that contains an SMD2 entry called 'my_smd_expt7'.

```
f['my_smd_expt7'] : group - this is the SMD2 entry
f['my_smd_expt7'].attrs['format'] = "SMD"
f['my_smd_expt7'].attrs['date_created'] = <time.ctime() upon creation>
f['my_smd_expt7'].attrs['date_modified'] = <time.ctime() upon last save>
f['my_smd_expt7'].attrs['description'] = "Description of the dataset goes here"

f['my_smd_expt7/data'] : group - this holds the data
f['my_smd_expt7/data/raw'] : dataset[int/float/double (nmol,ntime,ndata)] - this is the data for each molecule
<!-- f['my_smd_expt7/data/classes'] : dataset[int (nmol)] - specifies the class for each molecule -->
f['my_smd_expt7/data/source_index'] : dataset[int (nmol)] - specifies the source for each molecule

f['my_smd_expt7/sources'] : group - This holds the information about the origin of each molecule (e.g., separate movies)
f['my_smd_expt7/sources/0'] : group - Information about source 0 in the source_index list
f['my_smd_expt7/sources/0'] : group - Information about the Nth source in the source_index list
f['my_smd_expt7/sources/0'].attrs['source_name'] = "<Description of this source, e.g. movie 12">
f['my_smd_expt7/sources/0'].attrs['<Name for a bit of metadata for this source>'] = "<the metadata itself>"
f['my_smd_expt7/sources/0'].attrs['<Name for another bit of metadata for this source>'] = "<the metadata itself>"
...... more metadata for source 0
f['my_smd_expt7/sources/1'] : group - information about source 1 in the source_index list
.... entries for source 1
.. more sources

f['my_smd_expt7/analysisA'] : group  - this is an example of how you would save an analysis (i.e., within this hdf5 group)
f['my_smd_expt7/program7result'] : group - each program is responsible for the format/specification beyond this point...
```

## Questions
* What do you do if you want to combine different types of data into the same dataset (e.g., movie punchouts and XY coordinates)?
Use two separate SMDs, since the data is distinct. You can use the decription to reference the different entries
