package com.company;

import weka.classifiers.Classifier;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.converters.CSVLoader;

import java.io.File;
import java.io.IOException;
import java.util.Random;

public class Main {

    public static void main(String[] args) {
        //load data
        Instances data = loadData();

        //build classifier
        Classifier classifier = buildClassification(data);


        //classify a new instance
        try {
            for (int i = 0; i < data.numInstances(); i++) {
                double pred = classifier.classifyInstance(data.instance(i));
                System.out.print("ID: " + data.instance(i).value(0));
                System.out.print(", actual: " + data.classAttribute().value((int) data.instance(i).classValue()));
                System.out.println(", predicted: " + data.classAttribute().value((int) pred));
            }
        } catch (Exception e) {
            e.printStackTrace();
        }

    }


    public static Instances loadData() {
        try {
            CSVLoader loader = new CSVLoader();
            loader.setSource(new File("resources/sp2003.csv"));
            Instances data = loader.getDataSet();
            data.setClassIndex(data.numAttributes() - 1);
            return data;
        } catch (IOException e1) {
            e1.printStackTrace();
        }
        return null;
    }


    public static Classifier buildClassification(Instances data) {
        RandomForest rf = new RandomForest();
        try {
            rf.buildClassifier(data);
            Evaluation eval = new Evaluation(data);
            eval.crossValidateModel(rf, data, 10, new Random(1));
            System.out.println(eval.toSummaryString());
        } catch (Exception e) {
            e.printStackTrace();
        }
        return rf;
    }


}
