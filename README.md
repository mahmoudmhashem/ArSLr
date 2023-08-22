# ArSLr: Arabic Sign Language Recognition Repository

Welcome to the ArSLr repository! This repository is dedicated to the recognition of Arabic Sign Language gestures. Our project is aimed at building models that can accurately recognize various aspects of Arabic Sign Language, including alphabet letters and dynamic words. This README will guide you through the contents of the repository and provide instructions on how to use our models.

## Table of Contents

- [Introduction](#introduction)
- [Directory Structure](#directory-structure)
- [Alphabets Recognition](#alphabets-recognition)
- [Dynamic Words Recognition](#dynamic-words-recognition)
- [Demos](#demos)
- [Setup and Usage](#setup-and-usage)
- [Local Usage](#local-usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction

ArSLr stands for Arabic Sign Language Recognition, and our repository focuses on creating machine learning models capable of recognizing Arabic Sign Language gestures. We have divided our project into two main phases: **Alphabets Recognition** and **Dynamic Words Recognition**. Each phase has its own set of models, datasets, and interactive Colab notebooks for you to experiment with.

## Directory Structure

The repository is organized into the following directories:

- **alphabets**: This directory contains the first phase of our project, where we built a model capable of recognizing all 32 letters of Arabic Sign Language. We utilized a dataset collected in Emarat. Refer to the [alphabets README](alphabets/README.md) for more details.

- **dynamic_words**: In the second phase, we focused on recognizing dynamic words in Arabic Sign Language. We trained a model using a dataset collected at Mansoura University. Learn more about this phase in the [dynamic_words README](dynamic_words/README.md).

- **demos**: This directory includes demonstration videos showcasing the capabilities of both our alphabets and dynamic words recognition models.

## Alphabets Recognition

In the **alphabets** phase, we developed a model that can accurately recognize each of the 32 letters of Arabic Sign Language. Our model utilizes a pipeline of Mediapipe and Support Vector Machine (SVM), achieving an impressive accuracy of 95% on the test dataset. You can find more information about this phase in the [alphabets README](alphabets/README.md).

![Alphabets Demo](https://github.com/mahmoudmhashem/ArSLr/blob/master/demos/Letters-of-Arabic-Sign-Language_demo.gif)
## Dynamic Words Recognition

The **dynamic_words** phase focuses on recognizing dynamic words represented by sign gestures in Arabic Sign Language. Our model, powered by a combination of Mediapipe and Gated Recurrent Unit (GRU), achieves an accuracy of 97% on the test dataset. More details can be found in the [dynamic_words README](dynamic_words/README.md).

![Dynamic Words Demo](Dynamic-Words-of-Arabic-Sign-Language_demo.gif)

## Demos

The **demos** directory contains demonstration videos that showcase the capabilities of both our alphabets and dynamic words recognition models.

## Setup and Usage

To experience the power of our models, you can use the provided interactive Colab notebooks. Simply click the buttons below to launch the notebooks in Google Colab:

- Letters Demo: [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mahmoudmhashem/ArSLr/blob/master/alphabets/inference%20stage/test.ipynb)
- Dynamic Words Demo: [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mahmoudmhashem/ArSLr/blob/master/dynamic_words/inference%20stage/%5Brealtime%5D%20test_lstm_new_mediapipe.ipynb)

## Local Usage

If you prefer to try our models locally, follow these steps:

1. Clone this repository to your local machine.
2. Navigate to the respective phase directories: **alphabets** or **dynamic_words**.
3. Follow the instructions in the provided Jupyter notebook to set up and run the models locally.

## Contributing

We welcome contributions from the community! If you have ideas for improvements, bug fixes, or additional features, please open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

---

Feel free to explore the ArSLr repository and immerse yourself in the world of Arabic Sign Language recognition! If you have any questions or need assistance, please don't hesitate to reach out.

**Contacts:**
[LinkedIn](https://www.linkedin.com/in/mahmoudmahashem/)
[kaggle](https://www.kaggle.com/mahmoudmhashem)
