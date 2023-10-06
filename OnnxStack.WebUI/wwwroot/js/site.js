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