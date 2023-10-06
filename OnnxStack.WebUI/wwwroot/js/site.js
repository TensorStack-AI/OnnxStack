// Please see documentation at https://docs.microsoft.com/aspnet/core/client-side/bundling-and-minification
// for details on configuring this project to bundle and minify static web assets.

// Write your JavaScript code.
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