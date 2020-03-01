// tf.enableDebugMode();

async function main() {
  const model =await tf.loadLayersModel(
    "tensorflow_js_models/flickr_flower_photos_generator/model.json", { strict: false }
  );

  const inputCanvas      = document.getElementById('input');
  const predictionCanvas = document.getElementById("predicted");
  const edgesCanvas      = document.getElementById("edges");
  const webcamElement    = document.getElementById("webcam");

  const camera = await tf.data.webcam(webcamElement, {
    resizeWidth : 256,
    resizeHeight: 256
  });

  while (true) {
    const image = await camera.capture();
    await tf.browser.toPixels(image, inputCanvas);

    const ctx     = inputCanvas.getContext("2d");
    const imgData = ctx.getImageData(0, 0, inputCanvas.width, inputCanvas.height);
    const mat     = cv.matFromImageData(imgData);
    const dst     = new cv.Mat();
    cv.cvtColor(mat, mat, cv.COLOR_RGB2GRAY, 0);
    cv.Canny(mat, dst, 50, 100, 3, false);
    cv.imshow('edges', dst);

    const inputForModel = await tf.browser.fromPixels(edgesCanvas)
      .div(tf.scalar(127.5))
      .sub(tf.scalar(1.0))
      .expandDims(0);

    const predicted = model.predict(inputForModel);

    const inputForCanvas = predicted
      .squeeze()
      .add(1)
      .mul(127.5)
      .asType("int32");

    await tf.browser.toPixels(inputForCanvas, predictionCanvas);

    image.dispose();
    inputForModel.dispose();
    predicted.dispose();
    inputForCanvas.dispose();
    mat.delete();
    dst.delete();

    await tf.nextFrame();
  }
}

main();
