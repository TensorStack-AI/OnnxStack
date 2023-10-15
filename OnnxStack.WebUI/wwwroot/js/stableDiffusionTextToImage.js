const stableDiffusionTextToImage = () => {

    const LAYOUT_MAX_WIDTH = 1024;
    const LAYOUT_MAX_HEIGHT = 512;
    const LAYOUT_HISTORY_MAX_WIDTH = 256;
    const LAYOUT_HISTORY_MAX_HEIGHT = 256;

    let diffusionProcess;
    const buttonClear = $("#btn-clear")
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
            return;

        const schedulerParams = serializeFormToJson(schedulerParameterForm);
        if (!validateForm(schedulerParameterForm))
            return;

        processBegin();
        updatePlaceholderImage(true);
        diffusionProcess = await connection
            .stream("ExecuteTextToImage", promptParams, schedulerParams)
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
        buttonCancel.removeAttr("disabled");
        buttonExecute.attr("disabled", "disabled");
        promptParameterForm.find(".form-control, .slider").attr("disabled", "disabled");
        schedulerParameterForm.find(".form-control, .slider").attr("disabled", "disabled");
    }

    const processEnd = () => {
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

    // Map UI Events/Functions
    $(".image2image-control").hide();
    buttonCancel.on("click", cancelDiffusion);
    buttonClear.on("click", clearHistory);
    buttonExecute.on("click", async () => { await executeDiffusion(); });
    textBoxWidth.on("change", () => { updatePlaceholderImage(false); });
    textBoxHeight.on("change", () => { updatePlaceholderImage(false); }).trigger("change");

    // Map signalr functions
    connection.on("OnError", onServerError);
    connection.on("OnMessage", onServerMessage);
    connection.on("OnCanceled", onServerCanceled);
    connection.on("OnProgress", onServerProgress);
    connection.on("OnResponse", onServerResponse);
    connection.start();
}
