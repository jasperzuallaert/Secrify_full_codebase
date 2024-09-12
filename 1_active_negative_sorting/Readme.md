# Data processing for active negative sorting
Here, the code is found that was used to process the data files obtained from both the old and the new SECRiFY experiments.
## Requirements
* Python packages:
   * pandas
   * scikit-learn
   * matplotlib
   * seaborn
* CD-HIT

## Files
### v1_process_data
This notebook does the data processing, starting from the secretability read counts found in three source files:

* The enriched and depleted fragments from two replicates with active negative sorting
   * `original_data/enriched.txt` 
   * `original_data/depleted.txt`
* The previous _Pichia pastoris_ results table with read counts for three replicates
* `original_data/Pp_resultstable_enriched.txt`

It will process read counts, compute fold changes, eliminate redundancy and inconsistencies, and divide the final data into train/validation/test sets with sequences across datasets being less than 70% identical.

### naive_predictors.ipynb
This notebook contains code for naive predictors, that look only at amino acid frequency and sequence length.