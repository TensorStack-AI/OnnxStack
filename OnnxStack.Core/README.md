# OnnxStack.Core - Onnx Services for .NET Applications

OnnxStack.Core is a library that provides higher-level services for use in .NET applications. It offers extensive support for features such as dependency injection, .NET configuration implementations, ASP.NET Core integration, and IHostedService support.

## Getting Started

### .NET Core Registration

You can easily integrate `OnnxStack.Core` into your application services layer. This registration process sets up the necessary services and loads the `appsettings.json` configuration.

Example: Registering OnnxStack
```csharp
builder.Services.AddOnnxStack();
```

