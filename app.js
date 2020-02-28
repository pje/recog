import * as tf from "https://unpkg.com/@tensorflow/tfjs@1.5.2/dist/tf.esm.js?module";

async function main() {
  const model = tf.loadLayersModel(
    "tensorflowjs/models/flickr_flowers_AtoB_generator/model.json"
  );

  const predictionCanvas = document.getElementById("predicted");
  const webcamElement = document.getElementById("webcam");
  const camera = await tf.data.webcam(webcamElement, {
    resizeWidth: 256,
    resizeHeight: 256
  });

  while (true) {
    const image = await camera.capture();

    const inputForModel = image
      .div(127.5)
      .add(-1.0)
      .expandDims(0);

    // const predicted = model.predict(inputForModel);
    const predicted = inputForModel;

    const inputForCanvas = inputForModel
      .squeeze()
      .add(1)
      .mul(127.5)
      .asType("int32");

    await tf.browser.toPixels(inputForCanvas, predictionCanvas);

    image.dispose();
    inputForModel.dispose();
    predicted.dispose();
    inputForCanvas.dispose();

    await tf.nextFrame();
  }
}

main();
