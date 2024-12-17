# Zero Knowledge Verification of Machine Learning Robustness
## Introcution
This GitHub Repo contains the code for verify the robustness of a neural network. Each folder contains relevant code with this project. Below is the introduction for each folder in order appeared in the repo.
- Equal
  - This folder is a simple illustration of how to assign circuit, create witness, generate proof. It also shows the required addition files (go.sum and go.mod)
- ProofML
  - This is the main folder that contains the code for proving the robustness of a NN. All the source code is in the file **main.go**
- RNG
  - This file suppose to contain the random number generator. However, this due to the lack of modular arithmetic, this code doesn't quite work. There is existing zk RNG in this Github Repo: [randomina
](https://github.com/iluxonchik/randomina)
- ReadAndWrite
  - This folder contians the code for exporting the proof and verification key and read it in another folder, simulating the interaction between prover and verifier. In order to generate the proof and vk, use the Proof folder and use Verifier folder for read the proof and vk.
- ReadJson/One
  - This folder contains testing code for properly read json in golang.
- Sudoku
  - This folder contains the sudoku example.
- Test
  - This folder is used to test some codes and learn golang. 
