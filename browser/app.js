// tf.enableDebugMode();

async function main() {
  const STATE = {
    videoEnabled    : false,
    predictEnabled  : false,
    testImageEnabled: false,
    camera          : null
  };

  const videoEnabledCheckbox     = document.querySelector("#video_on");
  const predictEnabledCheckbox   = document.querySelector("#predict_on");
  const testImageEnabledCheckbox = document.querySelector("#test_image_on");

  const inputCanvas      = document.getElementById("input");
  const predictionCanvas = document.getElementById("predicted");
  const edgesCanvas      = document.getElementById("edges");
  const cameraElement    = document.getElementById("camera");
  const imageTestElement = document.getElementById("image_test");

  async function onVideoEnabledChange() {
    STATE.videoEnabled = !!videoEnabledCheckbox.checked;

    if (STATE.videoEnabled) {
      STATE.camera = await tf.data.webcam(cameraElement, {
        resizeWidth : 256,
        resizeHeight: 256
      });
    } else {
      STATE.camera.stop();
    }
  }

  const onpredictEnabledChange = () => {
    STATE.predictEnabled = !!predictEnabledCheckbox.checked;
  };

  const onTestImageEnabledChange = () => {
    STATE.testImageEnabled = !!testImageEnabledCheckbox.checked;
  };

  testImageEnabledCheckbox.addEventListener("change", onTestImageEnabledChange);
  videoEnabledCheckbox.addEventListener("change", onVideoEnabledChange);
  predictEnabledCheckbox.addEventListener("change", onpredictEnabledChange);

  const model = await tf.loadLayersModel(
    "tensorflow_js_models/flickr_flower_illustrations_generator/model.json",
    { strict: false }
  );

  await onVideoEnabledChange();
  await onpredictEnabledChange();
  await onTestImageEnabledChange();

  while (true) {
    let frameStart = new Date().getTime();
    // input image
    if (!!STATE.videoEnabled) {
      const image = await STATE.camera.capture();
      await tf.browser.toPixels(image, inputCanvas);
    } else if (!!STATE.testImageEnabled) {
      const imageTestElement = document.getElementById("image_test");
      inputCanvas.getContext("2d").drawImage(imageTestElement, 0, 0);
    }

    // edges image
    const ctx     = inputCanvas.getContext("2d");
    const imgData = ctx.getImageData(
      0,
      0,
      inputCanvas.width,
      inputCanvas.height
    );
    const mat = cv.matFromImageData(imgData);
    const dst = new cv.Mat();
    cv.cvtColor(mat, mat, cv.COLOR_RGB2GRAY, 0);
    cv.Canny(mat, dst, 50, 100, 3, false);
    cv.imshow("edges", dst);

    mat.delete();
    dst.delete();

    if (!!STATE.predictEnabled) {
      const inputForModel = await tf.browser
        .fromPixels(edgesCanvas)
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

      inputForModel.dispose();
      predicted.dispose();
      inputForCanvas.dispose();
    }

    await tf.nextFrame();

    let frameEnd        = new Date().getTime();
    let secondsForFrame = (frameEnd - frameStart) / 1000.0;
    let framerate       = (1 / secondsForFrame).toFixed(2);

    document.getElementById("framerate").innerHTML = `${framerate} fps`;
  }
}

main();
