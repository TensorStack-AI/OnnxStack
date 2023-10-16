const stableDiffusionImageInpaint = () => {

    const LAYOUT_MAX_WIDTH = 512;
    const LAYOUT_MAX_HEIGHT = 512;
    const LAYOUT_HISTORY_MAX_WIDTH = 256;
    const LAYOUT_HISTORY_MAX_HEIGHT = 256;

    let mask_canvas;
    let mask_ctx;
    let mask_flag = false;
    let mask_prevX = 0;
    let mask_currX = 0;
    let mask_prevY = 0;
    let mask_currY = 0;
    let mask_dot_flag = false;
    let mask_color = "black";
    let mask_size = 15;
    let diffusionProcess;
    const buttonClear = $("#btn-clear");
    const buttonCancel = $(".btn-cancel");
    const buttonExecute = $("#btn-execute");
    const textBoxWidth = $("#Width");
    const textBoxHeight = $("#Height");
    const promptParameterForm = $("#PromptParameters");
    const schedulerParameterForm = $("#SchedulerParameters");

    const outputContainer = $("#output-container");
    const outputResultTemplate = $("#outputResultTemplate").html();
    const progressResultTemplate = $("#progressResultTemplate").html();

    const outputHistoryContainer = $("#output-container-history");
    const historyOutputTemplate = $("#historyOutputTemplate").html();
    const historyProgressTemplate = $("#historyProgressTemplate").html();

    const inputContainer = $("#input-container");
    const inputResultTemplate = $("#inputResultTemplate").html();

    const connection = new signalR.HubConnectionBuilder().withUrl("/StableDiffusionHub").build();

    const onServerResponse = (response) => {
        if (!response)
            return;

        updateResultImage(response);
        processEnd();
    }

    const onServerError = (response) => {
        console.log("ERROR: " + response)
    }

    const onServerMessage = (response) => {
        console.log("MESSAGE: " + response)
    }

    const onServerProgress = (response) => {
        updateProgress(response);
    }

    const onServerCanceled = (response) => {
        updatePlaceholderImage();
        processEnd();
    }

    const executeDiffusion = async () => {
        const promptParams = serializeFormToJson(promptParameterForm);
        if (!validateForm(promptParameterForm))
            return; // TODO: display error

        const schedulerParams = serializeFormToJson(schedulerParameterForm);
        if (!validateForm(schedulerParameterForm))
            return; // TODO: display error

        const inputImageBase64 = getInputBase64();
        if (!inputImageBase64)
            return; // TODO: display error

        const maskImageBase64 = getInputMaskBase64();
        if (!maskImageBase64)
            return; // TODO: display error

        processBegin();
        updatePlaceholderImage(true);
        promptParams["inputImage"] = { imageBase64: inputImageBase64 };
        promptParams["inputImageMask"] = { imageBase64: maskImageBase64 };
        diffusionProcess = await connection
            .stream("ExecuteImageInpaint", promptParams, schedulerParams)
            .subscribe({
                next: onServerResponse,
                complete: onServerResponse,
                error: onServerError,
            });
    }

    const cancelDiffusion = async () => {
        diffusionProcess.dispose();
    }

    const updateResultImage = (response) => {
        const width = getWidth();
        const height = getHeight();

        addOutputResult(width, height, outputResultTemplate, response);
        outputHistoryContainer.find(".output-progress").remove();
        addOutputHistory(width, height, historyOutputTemplate, response);
    }

    const updatePlaceholderImage = (addToHistory) => {
        const width = getWidth();
        const height = getHeight();

        addOutputResult(width, height, progressResultTemplate)
        outputHistoryContainer.find(".output-progress").remove();
        if (!addToHistory)
            return;

        addOutputHistory(width, height, historyProgressTemplate);
    }

    const addInputResult = (width, height, template, data) => {
        const size = getSafeSize(width, height, LAYOUT_MAX_WIDTH, LAYOUT_MAX_HEIGHT);
        inputContainer.html(Mustache.render(template, {
            width: size.width,
            height: size.height,
            actualWidth: width,
            actualHeight: height,
            ...data
        }));
    }

    const addOutputResult = (width, height, template, data) => {
        const size = getSafeSize(width, height, LAYOUT_MAX_WIDTH, LAYOUT_MAX_HEIGHT);
        outputContainer.html(Mustache.render(template, {
            width: size.width,
            height: size.height,
            actualWidth: width,
            actualHeight: height,
            ...data
        }));
    }

    const addOutputHistory = (width, height, template, data) => {
        const size = getSafeSize(width, height, LAYOUT_HISTORY_MAX_WIDTH, LAYOUT_HISTORY_MAX_HEIGHT);
        outputHistoryContainer.prepend(Mustache.render(template, {
            width: size.width,
            height: size.height,
            actualWidth: width,
            actualHeight: height,
            ...data
        }));
    }

    const updateProgress = (response) => {
        const increment = Math.max(100 / response.total, 1);
        const progressPercent = Math.round(Math.min(increment * response.progress, 100), 0);
        const progressBar = $(".progress-result");
        progressBar.css("width", progressPercent + "%");
        progressBar.text(progressPercent + "%");
    }

    const processBegin = () => {
        $("#button-upload").attr("disabled", "disabled");
        buttonCancel.removeAttr("disabled");
        buttonExecute.attr("disabled", "disabled");
        promptParameterForm.find(".form-control, .slider").attr("disabled", "disabled");
        schedulerParameterForm.find(".form-control, .slider").attr("disabled", "disabled");
    }

    const processEnd = () => {
        $("#button-upload").removeAttr("disabled");
        buttonCancel.attr("disabled", "disabled");
        buttonExecute.removeAttr("disabled");
        promptParameterForm.find(".form-control, .slider").removeAttr("disabled");
        schedulerParameterForm.find(".form-control, .slider").removeAttr("disabled");
    }

    const clearHistory = () => {
        outputHistoryContainer.empty();
    }

    const getWidth = () => {
        return +$("option:selected", textBoxWidth).val();
    }

    const getHeight = () => {
        return +$("option:selected", textBoxHeight).val();
    }

    const clearInputImageUrl = () => {
        $("#img-input").removeAttr("src");
        addInputResult(getWidth(), getHeight(), inputResultTemplate, { imageUrl: '' });
    }

    const getInputBase64 = () => {
        const image = document.getElementById("img-input")
        const canvas = document.createElement("canvas");
        canvas.width = getWidth();
        canvas.height = getHeight();
        const ctx = canvas.getContext("2d");
        ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
        return canvas.toDataURL(); button - imgInpaint
    };

    $(document).on("click", "#button-upload", async function () {
        const width = getWidth();
        const height = getHeight();
        const result = await openModalGet(`/StableDiffusion?handler=UploadImage&Width=${width}&Height=${height}`);
        if (result.success) {
            addInputResult(width, height, inputResultTemplate, { imageUrl: result.base64, mask_size });
            mask_init();
            buttonExecute.removeAttr("disabled");
        }
    });

    $(document).on("click", "#button-transfer", async function () {
        const outputImageUrl = $("#img-result").attr("src");
        if (outputImageUrl) {
            addInputResult(getWidth(), getHeight(), inputResultTemplate, { imageUrl: outputImageUrl });
            buttonExecute.removeAttr("disabled");
        }
    });

    const setInitialImage = () => {
        const imageUrl = $("#InitialImage_Url").val();
        if (!imageUrl)
            return;

        const imageWidth = $("#InitialImage_Width").val();
        const imageHeight = $("#InitialImage_Height").val();
        textBoxWidth.val(imageWidth);
        textBoxHeight.val(imageHeight);
        addInputResult(getWidth(), getHeight(), inputResultTemplate, { imageUrl: imageUrl });
        mask_init();
        buttonExecute.removeAttr("disabled");
        history.pushState(null, "", location.href.split("?")[0]);
    }

    const mask_init = () => {
        mask_canvas = document.getElementById('img-mask');
        mask_ctx = mask_canvas.getContext("2d");
        mask_canvas.addEventListener("mousemove", function (e) { maskGetPosition('move', e) }, false);
        mask_canvas.addEventListener("mousedown", function (e) { maskGetPosition('down', e) }, false);
        mask_canvas.addEventListener("mouseup", function (e) { maskGetPosition('up', e) }, false);
        mask_canvas.addEventListener("mouseout", function (e) { maskGetPosition('out', e) }, false);
        $("#slider-mask-brush").on("input", function (e) { maskUpdateBrush($(this)); }).trigger("input");
        $("#btn-mask-clear").on("click", maskClear)
        $("#btn-mask-brush").on("click", maskEnablePaint);
        $("#btn-mask-eraser").on("click", maskEnableEraser);
        $(".mask-control").removeAttr("disabled");
    }

    const maskEnablePaint = () => {
        mask_color = "black";
        mask_ctx.globalCompositeOperation = "source-over";
        $("#btn-mask-brush").removeClass("btn-outline-info").addClass("btn-outline-success");
        $("#btn-mask-eraser").removeClass("btn-outline-success").addClass("btn-outline-info");
    }

    const maskEnableEraser = () => {
        mask_ctx.globalCompositeOperation = "destination-out";
        $("#btn-mask-brush").removeClass("btn-outline-success").addClass("btn-outline-info");
        $("#btn-mask-eraser").removeClass("btn-outline-info").addClass("btn-outline-success");
    }

    const maskUpdateBrush = (slider) => {
        const value = slider.val();
        slider.next().text(value);
        mask_size = value;
    }

    const maskDrawLine = () => {
        mask_ctx.beginPath();
        mask_ctx.moveTo(mask_prevX, mask_prevY);
        mask_ctx.lineTo(mask_currX, mask_currY);
        mask_ctx.strokeStyle = mask_color;
        mask_ctx.lineWidth = mask_size;
        mask_ctx.lineCap = 'round'
        mask_ctx.stroke();
        mask_ctx.closePath();
    }

    const maskGetPosition = (direction, e) => {
        if (direction == 'down') {
            mask_prevX = mask_currX;
            mask_prevY = mask_currY;
            mask_currX = e.clientX - mask_canvas.offsetLeft;
            mask_currY = e.clientY - mask_canvas.offsetTop;

            mask_flag = true;
            mask_dot_flag = true;
            if (mask_dot_flag) {
                mask_ctx.beginPath();
                mask_ctx.fillStyle = mask_color;
                mask_ctx.fillRect(mask_currX, mask_currY, 2, 2);
                mask_ctx.closePath();
                mask_dot_flag = false;
            }
        }
        if (direction == 'up' || direction == "out") {
            mask_flag = false;
        }
        if (direction == 'move') {
            if (mask_flag) {
                mask_prevX = mask_currX;
                mask_prevY = mask_currY;
                mask_currX = e.clientX - mask_canvas.offsetLeft;
                mask_currY = e.clientY - mask_canvas.offsetTop;
                maskDrawLine();
            }
        }
    }

    const maskClear = () => {
        mask_ctx.clearRect(0, 0, mask_canvas.width, mask_canvas.height);
    }

    const getInputMaskBase64 = () => {
        return mask_canvas.toDataURL();
    }


    // Map UI Events/Functions
    $(".image2image-control").hide();
    buttonCancel.on("click", cancelDiffusion);
    buttonClear.on("click", clearHistory);
    buttonExecute.on("click", async () => { await executeDiffusion(); });
    textBoxWidth.on("change", () => { updatePlaceholderImage(false); clearInputImageUrl(); });
    textBoxHeight.on("change", () => { updatePlaceholderImage(false); clearInputImageUrl(); }).trigger("change");

    setInitialImage();

    // Map signalr functions
    connection.on("OnError", onServerError);
    connection.on("OnMessage", onServerMessage);
    connection.on("OnCanceled", onServerCanceled);
    connection.on("OnProgress", onServerProgress);
    connection.on("OnResponse", onServerResponse);
    connection.start();
}