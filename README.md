# Dummy Model training with Polyaxon

## Train the Model Locally

```bash
#!/bin/bash
pip install -r requirements.txt
python ./src/train_log_reg.py --penalty={{ penalty }}
                      --C={{ C }}
                      --is_polyaxon_env={{ is_polyaxon_env }}
```

| Parameter               |  Description                                                                            | Valid Values | Default   |
| ---                     | ---                                                                                     | ---          | ---       |
| C                       | Intensity of regularisation                                                             | float        | 1.0       |
| penalty                 | Penalty to be used for regularisation                                                   | l1, l2       | l2        |
| is_polyaxon_env         | Indicate if running in Polyaxon                                                         | 0, 1         | 0         |

## Login to Polyaxon
```bash
pip install -U polyaxon-cli
polyaxon config set --host=****** --port=******
polyaxon login --username=****** --password=******
```
Validate you are logged in: `polyaxon cluster`


## Train the Model in Polyaxon

```bash
- create a project
`polyaxon project create --name=project-1`

- initialise the project
`polyaxon init project-1`

- Upload the code to polyaxon and run experiments
`polyaxon run -f polyaxonfiles/cpu.yml -u`

- See how much resourses experiment `3` is using:
`polyaxon experiment -xp 3 resources`

- Start a jupyter notebook
`polyaxon notebook start -f polyaxonfiles/notebook.yml`
```
