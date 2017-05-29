#!/bin/bash
python PCA_l2.py 'SRS.TotalRaw.Score' 0 && python PCA_l2.py 'SRS.TotalRaw.Score' 1 && python PCA_l2.py 'SRS.TotalRaw.Score' 2
python PCA_l2.py 'ADOS.Total' 1
