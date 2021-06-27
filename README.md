<!-- PROJECT LOGO -->
<br />
<p align="center">
<img  src="https://storage.googleapis.com/media.landbot.io/162368/channels/LAKCFRWSLYDP1HM2OWDPGY3XFUA9CR9F.png"  alt="Logo"  width="80"  height="80">
</a>
<h3 align="center">Vyno | Wine Recommendation System</h3>
<p align="center">
The Digital Sommelier.
<br />
</p>

<!-- TABLE OF CONTENTS -->
<details  open="open">
<summary>Table of Contents</summary>
<ol>
<li>
<a  href="#about-the-project">About The Project</a>
<ul>
<li><a  href="#built-with">Built With</a></li>
</ul>
</li>
<li>
<a  href="#getting-started">Getting Started</a>
<ul>
<li><a  href="#installation">Installation</a></li>
</ul>
</li>
<li><a  href="#usage">Usage</a></li>
<li><a  href="#roadmap">Roadmap</a></li>
<li><a  href="#contributing">Contributing</a></li>
<li><a  href="#license">License</a></li>
<li><a  href="#contact">Contact</a></li>
<li><a  href="#acknowledgements">Acknowledgements</a></li>
</ol>
</details>


  <!-- ABOUT THE PROJECT -->
## About The Project
Wine recommendation system build for [Vyno](https://www.vyno.ai).
### Built With
*  [Flask](https://flask.palletsprojects.com/en/2.0.x/)
*  [Gensim](https://radimrehurek.com/gensim/)
## Project Organization
```
├── LICENSE
├── README.md <- The top-level README for developers using this project.
│
├── models <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks <- Jupyter notebooks. Naming convention is a number (for ordering),
│
├── references <- Data dictionaries, manuals, and all other explanatory materials.
│
├── requirements.txt <- The requirements file for reproducing the analysis environment, e.g.
│
├── scripts <- Scripts used in this project
│
├── main.py <- Run the model
├── API.py <- Run API
```

<!-- GETTING STARTED -->
## Getting Started
To get a local copy up and running follow these simple example steps.
### Installation
1. Clone the repo.
```sh
git clone https://github.com/SebastiaanJohn/Vyno-Recommendation-Engine.git
```
2. Use the package manager [pip](https://pip.pypa.io/en/stable/) to install packages required.
```sh
pip install requirements.txt
```
<!-- USAGE EXAMPLES -->
## Usage
The project can be run in two different methods:
1. Through the **terminal**.
2. Through an **API**.

### 1. Terminal
Running the project through the terminal helps look at the results for different wines quickly and easily. Running the project is done as follows:
```bash
python3 main.py 'location/of/file.csv'
```
### 2. API
The API has two main calls, [/12wines, /2words], which can be interacted with by posting requests to the API. 

| The call to /12wines. 
The call is made by calling domain.com/12wines with the request's body consisting of a data frame of the 12 wines and their descriptions converted to a JSON object. This function will then return the eight descriptors from which the user should choose four. These descriptors need to be sent back to the API.

| The call to /2words. 
The call is made by calling domain.com/12wines with the body of the request consisting of the four chosen descriptors by the user and the data frame returned from the previous function. This function process the data frame and the four chosen descriptors then returns the six recommended wines based on the 12 wines originally from vyno and the two words chosen by the user.
The CSV must contain at least the following columns: NAME, DESCRIPTION. In addition, the column titles must be capitalized. See an example CSV below: 

| NAME                                        | DESCRIPTION                                                                                                                                                                                                                     |
|---------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Bella Modella, Primitivo, IGT Puglia, Italy | A very intense ruby red wine with garnet reflections, intense with hints of violet, black currant, plum jam and with light spicy notes. The taste is full and round thanks to the sweet tannins; it has hints of fruit compote. |

<!-- LICENSE -->
## License

Distributed under the MIT License.

<!-- CONTACT -->
## Contact

Sebastiaan Dijkstra - sebastiaandijkstra@gmail.com

Project Link: [https://github.com/SebastiaanJohn/Vyno-Recommendation-Engine](https://github.com/SebastiaanJohn/Vyno-Recommendation-Engine)

<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements
This project is built on the incredible work of [Roald Schuring](https://towardsdatascience.com/robosomm-chapter-3-wine-embeddings-and-a-wine-recommender-9fc678f1041e).