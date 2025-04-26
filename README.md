# Description
This is the open-source code of SAC-DSE.

# Start to Run
To run the programme, first modify the paths to the dataset and reports. The paths are specified by the files `config/config_ppo_xxx_xxx.yml` and `config/config_sac_xx_xxx.yml`.

Then, type the following command to run the programme:
```
python main.py --width_pref 5W_721
```
where `main.py` can be replaced by `main_4_MoPPO.py` or `BODKL_main.py` to run different DSE methods.

# Platform
We have run the programme on Windows 11, with 16GB RAM and Python ≥ 3.8.
