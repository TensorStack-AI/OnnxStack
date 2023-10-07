let token = $('input[name="__RequestVerificationToken"]').val();
$.ajaxPrefilter(function (options, originalOptions) {
    options.async = true;
    if (options.type.toUpperCase() == "POST") {
        options.data = $.param($.extend(originalOptions.data, { __RequestVerificationToken: token }));
    }
});
$.ajaxSetup({ cache: false });


const postJson = (url, vars) => {
    return $.ajax({
        url: url,
        cache: false,
        async: true,
        type: "POST",
        dataType: 'json',
        data: vars
    });
}


const getJson = (url, vars) => {
    return $.ajax({
        url: url,
        cache: false,
        async: true,
        type: "GET",
        dataType: 'json',
        data: vars
    });
}


const scaleFactor = (value) => {
    if (value == 512) {
        return 0.75;
    }
    else if (value == 576) {
        return 0.7;
    }
    else if (value == 640) {
        return 0.65;
    }
    else if (value == 704) {
        return 0.6;
    }
    else if (value == 768) {
        return 0.55;
    }
    else if (value == 832) {
        return 0.5;
    }
    else if (value == 896) {
        return 0.45;
    }
    else if (value == 960) {
        return 0.4;
    }
    else if (value == 1024) {
        return 0.35;
    }
    return 1;
}


const Enums = {
    ProcessResult: Object.freeze({
        Progress: 0,
        Completed: 1,
        Canceled: 2,
        Error: 10
    }),
    GetName: (enumType, enumKey) => {
        return Object.keys(enumType)[enumKey]
    },

    GetValue: (enumType, enumName) => {
        return enumType[enumName];
    }
};



const downscaleDelta = (value, maxValue) => {
    let newValue = value;
    if (value > maxValue) {
        while (newValue > maxValue) {
            newValue -= 16;
        }
        return value / newValue;
    }
    return 1;
}

const upscaleDelta = (value, maxValue) => {
    let newValue = value;
    if (value < maxValue) {
        while (newValue < maxValue) {
            newValue += 16;
        }
        return newValue / value;
    }
    return 1;
}

const getSafeSize = (width, height, widthMax, heightMax) => {
    if (width > widthMax || height > heightMax) {
        const widthDelta = downscaleDelta(width, widthMax);
        const heightDelta = downscaleDelta(height, heightMax);
        const delta = Math.max(widthDelta, heightDelta);
        return {
            width: width / delta,
            height: height / delta
        };
    }
    else if (width < widthMax || height < heightMax) {
        const widthDelta = upscaleDelta(width, widthMax);
        const heightDelta = upscaleDelta(height, heightMax);
        const delta = Math.min(widthDelta, heightDelta);
        return {
            width: width * delta,
            height: height * delta
        };
    }
    return {
        width: width,
        height: height
    };
}

const serializeFormToJson = (form) => {
    // Disable hidden checkbox fields
    form.find('[type=hidden]').each(function (i, field) {
        $(field).attr('disabled', "disabled");
    });
    const formDataJson = {};
    console.log(form.attr("id"))
    const formData = new FormData(document.getElementById(form.attr("id")));
    formData.forEach((value, key) => {

        if (key.includes("."))
            key = key.split(".")[1];

        // Convert number strings to numbers
        if (!isNaN(value) && value.trim() !== "") {
            formDataJson[key] = parseFloat(value);
        }
        // Convert boolean strings to booleans
        else if (value === "true" || value === "false") {
            formDataJson[key] = (value === "true");
        }
        else {
            formDataJson[key] = value;
        }
    });
    return formDataJson;
}

const validateForm = (form) => {
    form.validate();
    return form.valid();
}

$(".slider").on("input", function (e) {
    const slider = $(this);
    slider.next().text(slider.val());
}).trigger("input");

$(document).on("click", ".seed-host", function (e) {
    $("#Seed, .Seed").val($(this).text());
});