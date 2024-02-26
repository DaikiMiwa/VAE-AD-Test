# Instrattion & Requirements
We used Python 3.10.12. To run this package, please install dependencies with following code.

```
./install.sh
```

# Reproducibility
Since we have already got the results in advance, you can reproduce the figures by running following code. 
The results will be save in "/plot" folder.
```
cd plot
python plot.py
```

To reproduce the results, please see the following instructions　after a Instrattion step.
For the setting of one reference image, please run
All the results will be save the "./experiment_result" folder as csv file.

For the experiment of type I error control, please run
```
./experiment_typeIerror.sh
```

For the experiment of power, please run
```
./experiment_power.sh
```

For the experiment of robustness, please run
```
./experiment_robustness.sh
```