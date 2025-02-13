# Painting Tools and Dataset

![Paintout samples used in the Kubelka-Munk model](img/pigments_overview.png)
<small>_Processed paint samples, using the Kubelka-Munk model to compute reflection and the `painting_tools` package to display the RGB values. [See the corresponding notebook](https://github.com/rubenwiersma/painting_tools/blob/main/notebooks/02_oilpaint_dataset/02_captured_parameter_estimation.ipynb) to understand how these parameters were estimated._</small>

This repository contains code and links to data for painting analysis using hyperspectral data. This data can be used in technical art history and computer graphics applications, for example for pigment mapping and spectral upsampling. Code and data for this type of research is often hard to find and we hope that sharing this repository contributes to open code and data in technical art history.

The [accompaying data](#download-links) contains:
- Hyperspectral scans of [nine historical reconstructions](#paintings-and-pigments) by professional painters.
- Hyperspectral scans of [ten historical pigments](#paintings-and-pigments) in oilpaint.
- Several painting stages of one of the captured paintings (Vermeer's Milkmaid).

For each scan, we provide:
- Raw scan files from the Specim IQ hyperspectral scanner.
- [Code](https://github.com/rubenwiersma/painting_tools/tree/main/notebooks) to process the raw files (i.e., stitching scans that were made in segments, reading out spectra from samples).
- Processed data.
- An example python Notebook for unmixing paints with a reimplementation of [Pigmento](https://cragl.cs.gmu.edu/pigmento/) [[1](#references)] using a painting and paint database from our dataset.

The data was captured as part of research conducted at the [CGV group at the TU Delft](https://graphics.tudelft.nl) by [Ruben Wiersma](https://www.rubenwiersma.nl) for a project in collaboration with Elmar Eisemann and Adrien Bousseau. The reconstructions were painted by Lisa Wiersma, Charlotte Caspers, and Mané van Veldhuizen. Matthias Alfeld generously assisted in capturing the hyperspectral scans.

The data is shared under a permissive copyright license ([CC-BY-NC-SA](https://creativecommons.org/licenses/by-nc-sa/4.0/)) and the code under the MIT license. If this data or code is helpful to your research, please cite this repository and attribute the artists ([see below](#citing-and-attribution)) and [send me an email](mailto:rubenwiersma@gmail.com). I would love to hear your thoughts and feedback!

![Rachel Ruysch's bloemstilleven calibrated and stitched](img/ruysch_calibrated.png)
<small>_Left: The calibrated and stitched hyperspectral scan of Rachel Ruysch's bloemstilleven as reconstructed by Lisa Wiersma. Right: the RGB photograph used to register the hyperspectral scans. [See the corresponding notebook](https://github.com/rubenwiersma/painting_tools/blob/main/notebooks/01_painting_dataset/hyperspectral_calibration_stitching.ipynb) for more._</small>

## Table of contents
- [Download links](#download-links)
- [Paintings and pigments](#paintings-and-pigments)
- [How to use](#how-to-use)
- [Usecases](#usecases)
- [Frequently Asked Questions](#frequendly-asked-questions)

## Download links
- [Raw dataset (13,1GB)](https://drive.google.com/file/d/1lI4fgjCb7I2_HKqY6jzuolb4jTbVmwsE/view?usp=sharing)<br />Raw hyperspectral scans of nine paintings and paintout samples
- [Processed data (7,6GB)](https://drive.google.com/file/d/15IcxUPWeKb_jJdPlOT7n2WqnSUvuWyQo/view?usp=sharing)<br />Calibrated and stitched hyperspectral scans and reflectance measurements of paintout samples
- [Milkmaid process (1,2GB)](https://drive.google.com/file/d/1unbzny0CL96_2UvxIq6gLIA-2riDfujj/view?usp=sharing)<br />Images of painting stages for Vermeer's Milkmaid, photographed by Lisa Wiersma and registered and color-matched by Ruben Wiersma.

To use the data with our notebooks, extract each zip file into the same folder, e.g., `painting_data`. The folder structure should then look like:
```
painting_data
└─── painting_measurements
     |    735
     |    ...
     |    cropped
     |    visual
     |    LICENSE
     |    paintings.yaml
└─── paintout_measurements
     |    855
     |    ... 
     |    LICENSE
     |    paintouts.csv
└─── processed
     |    deheem_grapes_detail0
     |    ...
```

## Paintings and pigments
This dataset contains scans of reconstructions of the following paintings:
- Two details from [_Guirlande van vruchten met enkele bloemen_](https://www.mauritshuis.nl/ontdek-collectie/kunstwerken/49-guirlande-van-vruchten-met-enkele-bloemen) by Jan Davidsz de Heem (1650-1660)
<br />Reconstruction by Lisa Wiersma
- [_Trompe-l'oeil stilleven_](https://www.verenigingrembrandt.nl/nl/kunst/trompe-loeil-brievenbord-1) by Samuel van Hoogstraten (1664)
<br />Reconstruction by Lisa Wiersma. [Episode of _Het geheim van de meester_](https://www.avrotros.nl/archive/samuel-van-hoogstraten-brievenbord-trompe-loeil~fis8puty/) documenting the reconstruction process.
- [_Vaas met bloemen_](https://www.mauritshuis.nl/ontdek-collectie/kunstwerken/151-vaas-met-bloemen) by Rachel Ruysch (1700)
<br />Reconstruction by Lisa Wiersma. [Episode of _Het geheim van de meester_](https://www.avrotros.nl/archive/ruysch-vaas-met-bloemen~guvjlaq8/) documenting the reconstruction process.
- [_De piskijker_](https://www.lakenhal.nl/nl/collectie/b-1500) by Jan Steen (ca. 1663-1665)
<br />Reconstruction by Lisa Wiersma. [Episode of _Het geheim van de meester_](https://www.avrotros.nl/archive/het-geheim-van-de-meester-21~zbhleotg/) documenting the reconstruction process.
- [_Meisje met de Parel_](https://www.mauritshuis.nl/ontdek-collectie/kunstwerken/670-meisje-met-de-parel) by Johannes Vermeer (1665)
<br />Reconstruction by Charlotte Caspers. [Episode of _Het geheim van de meester_](https://www.avrotros.nl/archive/johannes-vermeer-meisje-met-de-parel-01-09-2016~dxlxb682/) documenting the reconstruction process.
- Two detail panels (containing multiple details) of [_Meisje met de Parel_](https://www.mauritshuis.nl/ontdek-collectie/kunstwerken/670-meisje-met-de-parel) by Johannes Vermeer (1665)
<br />Reconstruction by Mané van Veldhuizen.
- [_Het melkmeisje_](https://www.rijksmuseum.nl/nl/stories/thema/vermeer/story/alle-vermeers-het-melkmeisje) by Johannes Vermeer (ca. 1660)
<br />Reconstruction by Lisa Wiersma. [Episode of _Het geheim van de meester_](https://www.avrotros.nl/archive/het-geheim-van-de-meester-16~pietb3vp/) documenting the reconstruction process.

The following pigments were measured (six-digit code refers to Kremer pigment codes). These were selected based on their use in Vermeer's Girl with a Pearl Earring and to cover a wide gamut:
- Titanium wit	46200
- Bone black	47100
- Chalk white	58000
- French Ochre 40030
- French Ochre JALS 40060
- Lapis Lazuli from chile	10550
- Raw umber greenish	40612
- Red lake	36020
- Indigo	36000
- Jaune de réséda (yellow lacquer)

## How to use
This repository contains a python package, `painting_tools`, to work with spectral data (conversion from spectral to RGB, stitching) and for mixing and layering pigments using the Kubelka-Munk model. In addition, we share several Jupyter Notebooks to process and use the accompanying dataset.

To get started, install the python package and open one of the notebooks. An explanation of each notebook follows after the installation instructions.

### Installation
Create a conda environment (find instructions to [install conda here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)) with python 3.12 and activate the environment. Run the following commands in your terminal:
```bash
conda create -n painting_tools python=3.12
conda activate painting_tools
```
In your terminal, make sure that you are in the root directory of this repository and install the `painting_tools` package using pip:
```bash
pip install -e ./
```
This installs any required dependencies and the `painting_tools` package in the conda environment. The `-e` flag tells pip to install the package in the current folder, instead of the default conda folder. This is necessary to make sure that the measurement files included in the package are located correctly.

### Running the notebooks
We organized the notebook folders in three stages:
1. Calibration and stitching of the painting captures.
2. Reading out the paintouts and estimating Kubelka-Munk parameters and displaying datasets of pigments using `painting_tools`.
3. Example unmixing code with Pigmento.

You can run the notebooks using Jupyter. Make sure that your terminal is in the right directory (this repository, then `notebooks`) and run:
```bash
jupyter notebook
```
**Important** Make sure that you update the path to the dataset folder in the notebooks.

## Usecases
We share this data and code in the hopes that other researchers may find it useful. We make no claims about the validity of the data or the correctness of the code, but hope that it can inspire and inform researchers in technical art history and computer graphics on how to use and share data from technical inspections. Some example usecases of this dataset:
- Experiment with stitching algorithms for hyperspectral data.
- Identifying and mapping pigments in paintings using hyperspectral data (see notebook in `notebooks/03_example_unmixing`).
- High-dimensional data visualization and exploration.
- Algorithms for estimating Kubelka-Munk parameters and improved models.
- Improving ways to share and disemminate technical knowledge for painting analysis.

## Frequendly Asked Questions
- **When and why was this data captured?** The paintings were reconstructed for the TV show "Het geheim van de meester" by Lisa Wiersma. They were scanned in 2023 to support research in identifying and mapping pigments in historical paintings.
- **Why are you sharing this data?** I had planned to use the data for a research project, but was not able to use the data due to time limitations. I believe the data can be of use to other researchers, so why not share it? I also believe strongly that the field of technical art history needs more open source data and code. This is my humble contribution to this.
- **How were the paint samples prepared?** We followed the process for constructing paint samples described in the [thesis of Yoshi Okumura](https://repository.rit.edu/cgi/viewcontent.cgi?article=5896&context=theses). Each pigment was dissolved in linseed oil and ground to a workable paint. The paints were then mixed with Titanium white on a measuring scale so that we would get three mixtures: full pigment, 1:1 pigment:white, 1:2 pigment white. Please reach out to me if you would like more details on this process and the logs of the paint preparation. Technical details:
    - The pigments were bought from Kremer pigments.
    - Linseed oil (cooked) from Verfmolen de kat.
    - Substrate: Leneta opacity charts (2C)
    - Paintouts were applied using a draw bar with height 200 micrometer.
- **What data is known of the paint samples?** For each paint sample we recorded the following information (available in the file `paintouts.csv` in the raw paintouts data):
    - The name of the pigment and code in the Kremer pigments library.
    - The amount of pigment vs. linseed oil in grams (up to two decimal places).
    - The amount of paint (mixture of pigment and oil) in grams for each mixture.
    - A hyperspectral scan of the dried pigment.
- **What are the limitations of this data?** While we tried to capture and construct the data with utmost care, some flaws did arise:
    - The hyperspectral scan of Vermeer's Milkmaid was done through a glass pane. This was done to protect the frame. The implication is that this data is difficult to use for exact purposes, since the glass pane should be accounted for as an additional medium between the camera and the pigments.
    - The paintout samples were dried in a special drying room, but stacked before being completely dry. Some paper got stuck to the paint samples. To resolve this, we manually selected regions in the hyperspectral scans (`sample_locations_N.tif` in each raw data folder) that were not obstructed or damaged.
    - Our code assumes that the paintouts are opaque to estimate Kubelka-Munk parameters. While this is true for some paintouts, we can clearly observe effects from the substrate on some pigments.

## Citing and attribution
The dataset is licensed under CC BY-NC-SA 4.0. To view a copy of this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/. You can use this data for non-commercial projects and you must use proper attribution and share the new data with a similar license.

The paintings were painted by Lisa Wiersma and Mane van Veldhuizen (vermeer_girl_details) and captured by Ruben Wiersma and Matthias Alfeld. The paintouts were made by Ruben Wiersma and Mane van Veldhuizen and captured by Ruben Wiersma and Matthias Alfeld. 

If the code is useful in your projects, please cite this repository:
```
@software{Wiersma_Painting_2025,
  author = {Wiersma, Ruben},
  month = {02},
  title = {Painting Tools},
  url = {https://github.com/rubenwiersma/painting_tools},
  year = {2025}
}
```

## References
[1] J. Tan, S. DiVerdi, J. Lu and Y. Gingold, "Pigmento: Pigment-Based Image Analysis and Editing," in IEEE Transactions on Visualization and Computer Graphics, vol. 25, no. 9, pp. 2791-2803, 1 Sept. 2019, doi: 10.1109/TVCG.2018.2858238
