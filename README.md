# Logical Optimal Action

Logical Optimal Actions (LOA) is an action decision architecture of reinforcement learning applications with a neuro-symbolic framework which is a combination of neural network and symbolic knowledge acquisition approach for natural language interaction games. This repository has an implementation of LOA experiments consists of Python package on [TextWorld Commonsense (TWC) game](https://github.com/IBM/commonsense-rl). 

## Setup

- Anaconda 4.10.3
- Tested on Mac and Linux

```bash
git clone --recursive git@github.com:IBM/LOA.git loa
cd loa


# Setup games
git clone git@github.com:IBM/commonsense-rl.git 
cp -r commonsense-rl/games ./
rm -rf commonsense-rl


# Setup environment
conda create -n loa python=3.8
conda activate loa
conda install pytorch=1.10.0 torchvision torchaudio nltk=3.6.3 -c pytorch
pip install -r requirements.txt
python -m spacy download en


# Setup AMR Server: please follow (1) or (2) steps

# (1) If you have access to https://github.com/CognitiveHorizons/AMR-CSLogic/
cd third_party
git clone git@github.com:CognitiveHorizons/AMR-CSLogic.git amr-cslogic
cd amr-cslogic
# Execute installation scripts in INSTALLATION.md
export FLASK_APP=./amr_verbnet_semantics/web_app/__init__.py
python -m flask run --host=0.0.0.0 --port 5000 &
cd ../../

# (2) If you don't have access to the repo
mkdir -p cache
wget -O cache/amr_cache.pkl https://ibm.box.com/shared/static/klsvx54skc5wlf35qg3klo35ex25dbb0.pkl
# Note: This cache only contains sentences for "easy" game which is default in train.py
```

## Train and Test

```bash
python train.py

# if you have AMR server
python train.py --amr_server_ip localhost --amr_server_port 5000
```

## Citations

This repository provides code for the following paper, please cite the paper and give a star if you find the paper and code useful for your work.

- Daiki Kimura, Subhajit Chaudhury, Masaki Ono, Michiaki Tatsubori, Don Joven Agravante, Asim Munawar, Akifumi Wachi, Ryosuke Kohita, and Alexander Gray, "[LOA: Logical Optimal Actions for Text-based Interaction Games](https://aclanthology.org/2021.acl-demo.27/)", ACL-IJCNLP 2021.

  <details><summary>Details and bibtex</summary><div>

  The paper presents an initial demonstration of logical optimal action (LOA) on TextWorld (TW) Coin collector, TW Cooking, TW Commonsense, and Jericho. In this version, the human player can select an action by hand and recommendation action list from LOA with visualizing acquired knowledge for improvement of interpretability of trained rules.
  
  ```
  @inproceedings{kimura-etal-2021-loa,
      title = "{LOA}: Logical Optimal Actions for Text-based Interaction Games",
      author = "Kimura, Daiki  and  Chaudhury, Subhajit  and  Ono, Masaki  and  Tatsubori, Michiaki  and  Agravante, Don Joven  and  Munawar, Asim  and  Wachi, Akifumi  and  Kohita, Ryosuke  and  Gray, Alexander",
      booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing: System Demonstrations",
      month = aug,
      year = "2021",
      address = "Online",
      publisher = "Association for Computational Linguistics",
      url = "https://aclanthology.org/2021.acl-demo.27",
      doi = "10.18653/v1/2021.acl-demo.27",
      pages = "227--231"
  }
  ```
  </div></details>

### Applications for LOA

- Daiki Kimura, Masaki Ono, Subhajit Chaudhury, Ryosuke Kohita, Akifumi Wachi, Don Joven Agravante, Michiaki Tatsubori, Asim Munawar, and Alexander Gray, "[Neuro-Symbolic Reinforcement Learning with First-Order Logic](https://aclanthology.org/2021.emnlp-main.283/)", EMNLP 2021.

  <details><summary>Details and bibtex</summary><div>

  The paper shows an initial experiment of LOA by extracting first-order logical facts from text observation and external word meaning network on TextWorld Coin-collector. The experimental results show RL training with the proposed method converges significantly faster than other state-of-the-art neuro-symbolic methods in a TextWorld benchmark.

  ```
  @inproceedings{kimura-etal-2021-neuro,
      title = "Neuro-Symbolic Reinforcement Learning with First-Order Logic",
      author = "Kimura, Daiki  and  Ono, Masaki  and  Chaudhury, Subhajit  and  Kohita, Ryosuke  and  Wachi, Akifumi  and  Agravante, Don Joven  and  Tatsubori, Michiaki  and  Munawar, Asim  and  Gray, Alexander",
      booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
      month = nov,
      year = "2021",
      address = "Online and Punta Cana, Dominican Republic",
      publisher = "Association for Computational Linguistics",
      url = "https://aclanthology.org/2021.emnlp-main.283",
      pages = "3505--3511"
  }
  ```
  </div></details>


- Subhajit Chaudhury, Prithviraj Sen, Masaki Ono, Daiki Kimura, Michiaki Tatsubori, and Asim Munawar, "[Neuro-symbolic Approaches for Text-Based Reinforcement Learning](https://aclanthology.org/2021.emnlp-main.245/)", EMNLP 2021.

  <details><summary>Details and bibtex</summary><div>

  The paper presents SymboLic Action policy for Textual Environments (SLATE) method which is same concept of LOA. The method outperforms previous state-of-the-art methods for the coin collector game from 5-10x fewer training games.

  ```
  @inproceedings{chaudhury-etal-2021-neuro,
      title = "Neuro-Symbolic Approaches for Text-Based Policy Learning",
      author = "Chaudhury, Subhajit  and  Sen, Prithviraj  and  Ono, Masaki  and  Kimura, Daiki  and  Tatsubori, Michiaki  and  Munawar, Asim",
      booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
      month = nov,
      year = "2021",
      address = "Online and Punta Cana, Dominican Republic",
      publisher = "Association for Computational Linguistics",
      url = "https://aclanthology.org/2021.emnlp-main.245",
      pages = "3073--3078"
  }
  ```
  </div></details>


- Sarathkrishna Swaminathan, Dmitry Zubarev, Subhajit Chaudhury, Asim Munawar, “Reinforcement Learning with Logical Action-Aware Features for Polymer Discovery”, Reinforcement Learning for Real Life Workshop 2021.

  <details><summary>Details and bibtex</summary><div>

  The paper presents the first application of reinforcement learning in materials discovery domain that explicitly considers logical structure of the interactions between the RL agent and the environment. 

  ```
  @conference{swaminathan-etal-2021-reinforcement,
      title = "Reinforcement Learning with Logical Action-Aware Features for Polymer Discovery",
      author = "Swaminathan, Sarathkrishna  and  Zubarev, Dmitry  and  Chaudhury, Subhajit  and  Munawar, Asim",
      booktitle = "Reinforcement Learning for Real Life Workshop",
      year = "2021"
  }
  ```
  </div></details>


## License

MIT License
