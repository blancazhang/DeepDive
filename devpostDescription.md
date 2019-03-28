# Deep Dive Devpost

## Inspiration
Hate speech is present everywhere and with social media so readily available, people are able to hide behind a screen and feel less responsible for their language. This is indicative of a person who will likely create a hostile workplace, making other employees feel unsafe. We wanted to create a way for HR departments to proactively work against this happening and so- Deep Dive!

## What it does
Hate speech is present everywhere and with social media so readily available, people are able to hide behind a screen and feel less responsible for their language. This is indicative of a person who will likely create a hostile workplace, making other employees feel unsafe. We wanted to create a way for HR departments to proactively work against this happening and so- Deep Dive!

## How we built it
The ML algorithm responsible for Deep Dive was built with keras in python, and was trained on a dataset containing tweets labeled as either containing hate speech, offensive language, or neither. These tweets were then converted into vectors using GloVe (a pretrained word embedding model which was trained on tweets). We built an iOS app in Swift. The web app was built using HTML5/CSS3 and Bootstrap. Visuals and branding were rendered with the Adobe Creative Cloud suite.

## Challenges we ran into
Deploying the machine learning model to be used in our app, scraping tweets from specific users, finding twitter accounts with examples of hateful tweets to test our model on for development, converting ML models to various formats, not having the latest versions of software, and more

## Accomplishments that we're proud of
We have a model which we built ourselves front to end, and achieved 87% accuracy, well above the 33% random baseline, and comparative with similar state of the art models, We also have both a web and iOS application which is exciting. We had a team with an incredibly diverse range of experiences and expertise which allowed us to separate the tasks according to our strengths, and ultimately put our skills together to create an MVP.

## What we learned
We learned how to work as a team by dividing work and working separately- collaboration does not always mean working on the same thing at the same time. We learned new tools and functionalities such as coremltools and the Twitter API.

## What's next for DeepDive
Integrating the model fully with the iOS and web app we developed, loading live data from twitter users, fine tuning the model to classify the type of hate speech (sexism, racism, etc.), attempting other model implementations to improve accuracy, publish iOS app on store, create business model to evaluate potential use for real businesses

## Built with
Python, Jupyter Notebooks, JSON, Swift, Xcode, CoreML, Open Source Data, Twitter APIs, Bootstrap, HTML5, CSS3
Machine Learning, Natural Language Processing, Artificial Intelligence

## Devpost link
https://devpost.com/software/deep-dive

## Site Link
https://devpost.com/software/deep-dive

## GitHub Repo
https://github.com/tnmcneil/DeepDive

Stay tuned for more...