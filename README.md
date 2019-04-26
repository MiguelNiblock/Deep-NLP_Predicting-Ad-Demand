# Proposal- Predicting Ad Demand
---

## 1. Problem

When selling used goods online, a combination of tiny, nuanced details in a product description can make a big difference in drumming up interest.
Details like:

![image](proposal_images/image.png)

And, even with an optimized product listing, demand for a product may simply not exist–frustrating sellers who may have over-invested in marketing.

Websites for classified advertisements such as Craigslist are deeply familiar with this problem. Sellers may not make the most out of their listings. When a listing has little demand, it may be due to listing deficiencies (bad images, or bad description) or to the product itself (item not historically in demand). On the other hand, a listing with high demand might be underpriced, meaning less profit for the seller.

## 2. Solution

I propose a neural network model that predicts demand for an online advertisement based on its full description (title, description, images, etc.), its context (geographically where it was posted, similar ads already posted) and historical demand for similar ads in similar contexts. This model would inform sellers on how to best optimize their listing and provide some indication of how much interest they should realistically expect to receive.

### 2.1 Challenges

The most challenging part of this project would be dealing with the different kinds of data. NLP would be necessary for parsing titles and descriptions. Aside from common words and possibly LSA components, NLP features might include the count of misspelled words, or the number of total words. Object detection algorithms might identify poor image quality and other problematic cases, such as whether the objects appear too small in the frame (As a consequence of the camera being too far). And image contrast could measure the visibility of products. 

The total size of all data included is 123GB, of which only 2GB are csv files and the rest are images. This presents an obvious challenge. On one hand, using google collab would be convenient for GPU acceleration. On the other hand, that much data isn't storable with a free Google Drive account. Working locally might solve the size issue at the expense of slowing graphics processing. Perhaps we could limit the data to a workable compromise. 

### 2.2 Specializations Used

**NLP**

NLP's most important contribution lies be in the form of text summarization. Complete listings have a description field where the user freely writes about the product or service. Text extraction in the form of TFIDF components can help the model understand some meaningful patterns in the data.

There is a lot of text cleaning to do with this dataset. NLP will be incredibly useful for getting the most out of the variables `param_1, param_2, param_3` which are subcategories of each listing that have been provided by the user and are non-standardized.

Generating and filtering NLP features will be challenging because of the 1.5 Million datapoints and the sheer number of unique values per variable. Dummies will have to be evaluated for importance during the feature generation process, in order to avoid maxing out RAM memory.

**Keras and Deep Learning**

Having so much image data to play with, we'll naturally compare various forms of Neural Networks on the image data. 

## 3. Use Case

This prediction can be used to inform sellers on how to best optimize their listing and provide some indication of how much interest they should realistically expect to receive. Some online retailers already have implementations of similar models to make suggestions that help sellers. 

Ebay is a notable example of a similar implementation, which gives the seller an "Average Price Range". However, Ebay's implementation only relies on the product title, and only works for common items with plenty of demand such as iPhones and other popular products. The model proposed here would take into account the quality of the images uploaded and the quality of the product description in addition to the context and historical demand for similar products in the platform.

>*An implementation like this is equal to Craigslist on steroids.*

### 3.1 End-User Vision

When users create ads, they would receive recommendations based on the information in their listing as a whole. These recommendations are aimed at improving their experience of placing ads in the site, and therefore prevent them from migrating to other platforms. Based on all the entries a user has made to a listing, they'd be able to see the likelihood of their ad becoming a sale.

## 4. Milestones

The goal of this project is to achieve the best balance between predictive and explanatory power while reducing computational complexity. Good models which go into production are robust, which means their performance is stable under various conditions, and they aren't unnecessarily complex. Although platforms with GPU acceleration are available, RAM memory is still a limitation when loading such large datasets and performing transformations. This model should make the most out of the amount of memory provided by Kaggle, which is 13GB, and at the same time produce the lowest possible error along with an understanding of the sources for error.

## 5. Data

Avito is a Russian classified advertisements platform similar to Craigslist. They released a dataset including listings with fields such as title, description, location(city & region), price and images. The target variable is the Deal Probability, which is the likelihood that an ad actually sold something. It's not possible to verify every transaction with certainty, so this column's value can be any float from zero to one. In addition, there is supplemental historical data on every listing ID which shows the dates every ad was active on the site.

## 6. Short EDA
