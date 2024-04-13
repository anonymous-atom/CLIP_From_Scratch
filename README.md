<p align="center">
  <img src="https://images.openai.com/blob/fbc4f633-9ad4-4dc2-bd94-0b6f1feee22f/overview-a.svg" width="288" />
</p>
<p align="center">
    <h1 align="center">CLIP FROM SCRATCH</h1>
</p>
<p align="center">
    <em>Integrating Vision and Language</em>
</p>
<p align="center">
	<img src="https://img.shields.io/github/last-commit/anonymous-atom/CLIP_From_Scratch?style=flat&logo=git&logoColor=white&color=0080ff" alt="last-commit">
	<img src="https://img.shields.io/github/languages/top/anonymous-atom/CLIP_From_Scratch?style=flat&color=0080ff" alt="repo-top-language">
<p>
<p align="center">
		<em>Developed with :</em>
</p>
<p align="center">
	<img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white" alt="PyTorch">	
</p>
<hr>

##  Quick Links

> - [ Repository Structure](#-repository-structure)
> - [ Modules](#-modules)
> - [ Getting Started](#-getting-started)
>   - [ Installation](#-installation)
> - [ Contributing](#-contributing)
> - [ TODO ](#todo)
---

### TODO

- [x] Text Encoder
- [x] Vision Encoder
- [x] Projection Matrix
- [x] Basic Training Script
- [ ] Make Model GPU Compatible (Just little changes in Text Encoder Part)
- [ ] Write Distributed training scripts
- [ ] Add options to switch to custom text and vision encoder


---

##  Repository Structure

```sh
└── CLIP_From_Scratch/
    ├── CLIP.py
    ├── encoder.py
    ├── projection_head.py
    └── train.py
```

---

##  Modules


| File                                                                                                     | Summary                                                                                                                                                                               |
| ---                                                                                                      | ---                                                                                                                                                                                   |
| [encoder.py](https://github.com/anonymous-atom/CLIP_From_Scratch/blob/master/encoder.py)                 | The `encoder.py` provides text and image embeddings functionality. |
| [train.py](https://github.com/anonymous-atom/CLIP_From_Scratch/blob/master/train.py)                     | The train.py script trains a CLIP model, managing data loading, training steps, updating metrics, and learning rate retrieval.                              |
| [CLIP.py](https://github.com/anonymous-atom/CLIP_From_Scratch/blob/master/CLIP.py)                       | CLIP.py defines a multimodal CLIP model, integrating image and text encoders with projection heads for joint embedding. |
| [projection_head.py](https://github.com/anonymous-atom/CLIP_From_Scratch/blob/master/projection_head.py) | The `projection_head.py` defines a neural network module for transforming embeddings     |

---

##  Getting Started

###  Installation

1. Clone the CLIP_From_Scratch repository:

```sh
git clone https://github.com/anonymous-atom/CLIP_From_Scratch
```

2. Change to the project directory:

```sh
cd CLIP_From_Scratch
```

3. Install the dependencies:

```sh
pip install -r requirements.txt
```

---

---

##  Contributing

Contributions are welcome! Here are several ways you can contribute:

- **[Submit Pull Requests](https://github.com/anonymous-atom/CLIP_From_Scratch/blob/main/CONTRIBUTING.md)**: Review open PRs, and submit your own PRs.
- **[Join the Discussions](https://github.com/anonymous-atom/CLIP_From_Scratch/discussions)**: Share your insights, provide feedback, or ask questions.
- **[Report Issues](https://github.com/anonymous-atom/CLIP_From_Scratch/issues)**: Submit bugs found or log feature requests for Clip_from_scratch.

<details closed>
    <summary>Contributing Guidelines</summary>

1. **Fork the Repository**: Start by forking the project repository to your GitHub account.
2. **Clone Locally**: Clone the forked repository to your local machine using a Git client.
   ```sh
   git clone https://github.com/anonymous-atom/CLIP_From_Scratch
   ```
3. **Create a New Branch**: Always work on a new branch, giving it a descriptive name.
   ```sh
   git checkout -b new-feature-x
   ```
4. **Make Your Changes**: Develop and test your changes locally.
5. **Commit Your Changes**: Commit with a clear message describing your updates.
   ```sh
   git commit -m 'Implemented new feature x.'
   ```
6. **Push to GitHub**: Push the changes to your forked repository.
   ```sh
   git push origin new-feature-x
   ```
7. **Submit a Pull Request**: Create a PR against the original project repository. Clearly describe the changes and their motivations.

Once your PR is reviewed and approved, it will be merged into the main branch.

</details>


---



[**Return**](#-quick-links)

---
