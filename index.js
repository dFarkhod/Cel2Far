async function PredictCelciusToFarhrenheit() {

    // chiziqli regressiya (linear regression) uchun model quramiz:
    const model = tf.sequential();
    model.add(tf.layers.dense({ units: 1, inputShape: [1] }));

    // modelimizni kompilyatsiya qilamiz:
    model.compile({ loss: 'meanSquaredError', optimizer: 'sgd', metrics: 'accuracy' });

    // Modelimizni o'qitish (training) uchun mo'ljallangan kiruvchi va natijaviy ma'lumot:
    const data = tf.tensor2d([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [11, 1]);
    const labels = tf.tensor2d([32, 33.8, 35.6, 37.4, 39.2, 41, 42.8, 44.6, 46.4, 48.2, 50], [11, 1]);
     
    // Yuqoridagi ma'lumotni modelimizga kiritib, modelni o'qitamiz (training):
    const modelResult = await model.fit(data, labels, {
        epochs: 800, // davrlar soni - ya'ni modelni necha marta o'qitish soni
        callbacks: { onEpochEnd: (epoch, logs) => console.log(logs.loss) } 
        // har bir o'qitish davri tugaganida (onEpochEnd xodisasi) konsolga o'qitishning natijasidagi xatolar
        // miqdori chiqariladi...e'tibor bering xato miqdori borgan sayin kamayib boradi.
        }
    );

    // Endi kirishda 15 soni kiritilsa, natijada qanday son bo'lishini modelimiz bashorat qilib berishi kerak.
    // Ya'ni, harorat tselziy bo'yicha 15 daraja bo'lganida, farengeytda u qancha bo'lishini modelimiz bizga aniqlab beradi:
    const predictionResult = model.predict(tf.tensor2d([15], [1, 1]));
    predictionResult.print();
}


PredictCelciusToFarhrenheit();