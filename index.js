//const tf = require('@tensorflow/tfjs');

async function GetPrediction() {
   
    // Define a model for linear regression.
    const model = tf.sequential();
    model.add(tf.layers.dense({units: 1, inputShape: [1]}));
    
    model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});
    
    // Generate some synthetic data for training.
    const data = tf.tensor2d([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [11, 1]);
    const labels = tf.tensor2d([32, 33.8, 35.6, 37.4, 39.2, 41, 42.8, 44.6, 46.4, 48.2, 50], [11, 1]);
    
    // Train the model using the data.
    const modelResult = await model.fit(data, labels, {epochs: 1200});
    //console.log(modelResult.history.loss[0]);   
    
    const predictionResult = model.predict(tf.tensor2d([15], [1, 1]));
    //const result = model.predict(tf.tensor2d([15, 16, 17], [3, 1]));
    predictionResult.print();
}

GetPrediction();
