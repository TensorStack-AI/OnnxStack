const stableDiffusionImageToImage = () => {

    const LAYOUT_MAX_WIDTH = 512;
    const LAYOUT_MAX_HEIGHT = 512;
    const LAYOUT_HISTORY_MAX_WIDTH = 256;
    const LAYOUT_HISTORY_MAX_HEIGHT = 256;

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
            return;// TODO: display error

        const schedulerParams = serializeFormToJson(schedulerParameterForm);
        if (!validateForm(schedulerParameterForm))
            return;// TODO: display error

        const inputImageBase64 = getInputBase64();
        if (!inputImageBase64)
            return; // TODO: display error

        processBegin();
        updatePlaceholderImage(true);
        promptParams["inputImage"] = { imageBase64: inputImageBase64 };
        diffusionProcess = await connection
            .stream("ExecuteImageToImage", promptParams, schedulerParams)
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
        return canvas.toDataURL();
    };

    $(document).on("click", "#button-upload", async function () {
        const width = getWidth();
        const height = getHeight();
        const result = await openModalGet(`/StableDiffusion?handler=UploadImage&Width=${width}&Height=${height}`);
        if (result.success) {
            addInputResult(width, height, inputResultTemplate, { imageUrl: result.base64 });
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
        buttonExecute.removeAttr("disabled");
        history.pushState(null, "", location.href.split("?")[0]);
    }



    // Map UI Events/Functions
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