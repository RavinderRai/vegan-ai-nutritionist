# Notebooks

This folder contains notebooks where initial exploration was done. For the fully functioning refactored code see the modules folder.

To use these notebooks, you can reuse the poetry environments defined in the respective modules folders, i.e., to run the data_ingstion.ipynb notebook, navigate to the modules/data/ingestion/ folder in a terminal and run 

```bash
poetry shell
```

to start the environment. Now make sure `notebook` and `ipykernel` are installed in your poetry env, and then execute the command:

```bash
poetry run python -m ipykernel install --user --name=poetry-env --display-name "replace_with_env_name" 
```

Next run `jupyter notebook` to get a localhost link, and click on the `Select Kernel` button at the top right. Finally, select `Existing Jupyter Server...` and copy and paste the localhost link in there and hit enter.
