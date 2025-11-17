# Data Directory

## Structure

```
data/
├── species_1/
│   ├── chr1.fna
│   └── chr2.fna
└── species_2/
    └── genome.fa
```

## Requirements

- Format: FASTA (.fna, .fa, .fasta)
- Content: DNA sequences (A, C, G, T)
- Organization: One directory per species

## Data Sources

- NCBI: https://www.ncbi.nlm.nih.gov/genome/
- Ensembl Plants: https://plants.ensembl.org/
- Phytozome: https://phytozome-next.jgi.doe.gov/

## Adding Species

1. Create species directory in data/
2. Add FASTA files to directory
3. Update species_distribution in plant_lm_train.py Config class

Example:
```python
species_distribution = {
    "glycine_max": 1.0,
    "arabidopsis": 0.8,
    "your_new_species": 0.7,
}
```
