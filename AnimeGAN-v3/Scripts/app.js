(() => {
    async function init() {
        const pickerOpts = {
            types: [
                {
                    description: "Images",
                    accept: {
                        "image/*": [".png", ".jpg", ".jpeg"],
                    },
                },
            ],
            excludeAcceptAllOptions: true,
            multiple: false,
        };
        let modSe = document.querySelectorAll(".modelGridSec");
        let tfliteModel;
        let openFile = document.querySelector("#openFile");
        let genBtn = document.querySelector("#genBtn");
        let dwnBtn = document.querySelector("#dwnBtn");
        let imgCnt = document.querySelector("#imgCnt");
        let ldMod = document.querySelector("#loadModel");
        let dwnMes = document.querySelector("#dwnMes");
        let cnvCnt = document.querySelector("#cnvCnt");
        let runTxt1 = "<p>Image generation time will depend on your device performance.</p>";
        let runTxt2 = "<p>Running Model...</p>";
        cnvCnt.innerHTML = runTxt1;
        let im, canvas, ctx, model;
        modSe.forEach(elem => {
            elem.addEventListener("click", () => {
                elem.children[0].checked = true;
                model = elem.children[0].getAttribute("model");
                ldMod.disabled = false;
            })
        });
        ldMod.addEventListener("click", () => {
            let rp = new Promise((resolve, reject) => {
                dwnMes.classList.remove("hidden");
                ldMod.classList.add("hidden");
                setTimeout(() => {
                    resolve();
                }, 500);

            });
            rp.then(() => {
                tflite.loadTFLiteModel(`./AnimeGAN-v3/models/AnimeGANv3_${model}.tflite`).then(mode => {
                    tfliteModel = mode;
                    dwnMes.innerHTML = "Model Loaded";
                    setTimeout(() => {
                        dwnMes.classList.add("hidden");
                        ldMod.classList.remove("hidden");
                        dwnMes.innerHTML = "Downloading Model...";
                    }, 2000);
                });
            })
        })
        openFile.addEventListener("click", async () => {
            try {
                if (model) {
                    window.showOpenFilePicker(pickerOpts).then(async handle => {
                        handle[0].getFile().then(image => {
                            let urlw = URL.createObjectURL(image);
                            if (!im) {
                                imgCnt.innerHTML = "";
                                im = new Image();
                                imgCnt.appendChild(im);
                                im.style.position = "absolute";
                            }
                            im.src = urlw;
                            im.onload = () => {
                                let resol = im.width / im.height;
                                if (resol < 1) {
                                    let zoom = (imgCnt.clientHeight - 10) / im.height;
                                    im.style.zoom = zoom;
                                } else {
                                    let zoom = (imgCnt.clientWidth - 10) / im.width;
                                    im.style.zoom = zoom;
                                }
                                dwnBtn.classList.add("hidden");
                                genBtn.disabled = false;
                                genBtn.addEventListener("click", () => {
                                    let prom = new Promise((resolve, reject) => {
                                        genBtn.disabled = true;
                                        cnvCnt.innerHTML = runTxt1 + runTxt2;
                                        if (canvas) {
                                            cnvCnt.classList.remove("hidden");
                                            canvas.classList.add("hidden");
                                            ctx.clearRect(0, 0, canvas.width, canvas.height);
                                        };
                                        setTimeout(() => {
                                            resolve();
                                        }, 1000);
                                    });
                                    prom.then(() => {
                                        predict(im);
                                    })
                                });
                                URL.revokeObjectURL(urlw);
                            }
                        })
                    })
                } else {
                    window.alert("Load Model First!!!");
                }
            } catch (error) {
                console.error(error);
            }
        });
        async function getSaveHandle(name) {
            const opts = {
                suggestedName: name,
                types: [
                    {
                        description: "Image",
                        accept: {
                            'Image/*': [".png"],
                        },
                    },
                ],
            };
            let handle = await window.showSaveFilePicker(opts);
            return handle;
        };
        function writeFile(handle, url) {
            fetch(url).then(file=>{
                URL.revokeObjectURL(url);
                handle.createWritable().then(async (wt)=>{
                    await file.body.pipeTo(wt);
                });
            });
        };
        function predict(imag) {
            const finalTensor = tf.tidy(() => {
                let img = tf.browser.fromPixels(imag);
                let img2 = tf.image.resizeBilinear(img, [512, 512]);
                let img3 = tf.sub(tf.div(tf.expandDims(img2), 127.5), 1);
                console.log("Running Model...");
                let output = tfliteModel.predict(img3);
                return tf.mul(tf.add(output, 1), 127.5);
            });
            const rgb = Array.from(finalTensor.dataSync());
            const rgba = [];
            for (let i = 0; i < rgb.length / 3; i++) {
                for (let j = 0; j < 3; j++) {
                    rgba.push(rgb[i * 3 + j]);
                }
                rgba.push(255);
            };
            const imageData = new ImageData(Uint8ClampedArray.from(rgba), 512, 512);
            const resol = imag.height / imag.width;
            const pCan = document.createElement("canvas");
            pCan.width = 512;
            pCan.height = 512;
            const pCtx = pCan.getContext("2d");
            pCtx.putImageData(imageData, 0, 0);

            pCan.toBlob((blob) => {
                let url = URL.createObjectURL(blob);
                let imaf = new Image(512, resol * 512);
                imaf.src = url;
                imaf.onload = () => {
                    if (!canvas) {
                        canvas = document.createElement("canvas");
                        ctx = canvas.getContext("2d");
                        cnvCnt.parentElement.insertBefore(canvas, cnvCnt);
                    }
                    cnvCnt.classList.add("hidden");
                    canvas.classList.remove("hidden");
                    canvas.width = 512;
                    canvas.height = resol * 512;
                    ctx.drawImage(imaf, 0, 0, 512, resol * 512);
                    URL.revokeObjectURL(url);
                    genBtn.disabled = false;
                    dwnBtn.classList.remove("hidden");
                    dwnBtn.addEventListener("click", () => {
                        canvas.toBlob(async (blob) => {
                            let urle = URL.createObjectURL(blob);
                            let date = new Date();
                            let name = `AnimeGANv3-prediction-${date.getTime()}.png`;
                            let svHndl = await getSaveHandle(name);
                            writeFile(svHndl, urle);
                        });
                    });
                };
            });
        };
    };
    init();
})();