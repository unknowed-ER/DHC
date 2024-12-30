conda env create --name tf2sks --file=requirements.yml
conda activate tf2sks

pip install --upgrade tensorflow-gpu
pip install --upgrade tensorflow-probability
pip install better-exceptions colorlog colorful more_itertools parlai tensorflow-addons tf-geometric
sudo apt install openjdk-11-jdk libxml-parser-perl
pip install git+https://github.com/bckim92/language-evaluation.git
python -c "import language_evaluation; language_evaluation.download('coco')"

pip install --upgrade pip
pip install namedlist
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.0/en_core_web_sm-2.2.0.tar.gz
pip install tensorflow-gpu==2.0.0 tensorflow-probability==0.8.0 tensorflow-addons==0.6.0 torch==1.3.1 numpy==1.16.2 h5py tqdm better-exceptions colorlog colorful scikit-learn git+https://github.com/bckim92/language-evaluation.git spacy==2.1.8 nltk pylint pycodestyle mypy grip more_itertools isort pudb jupyter gast==0.2.2
pip install git+https://github.com/facebookresearch/ParlAI.git@51eada993206f5a5a264288acbddc45f33f219d8
pip install git+https://github.com/rsennrich/subword-nmt.git@18a5c87046d15290a1b7d947449052aa6d2b47cc
pip install pandas tf-geometric
pip install --upgrade protobuf==3.20.1
conda install ruamel.yaml