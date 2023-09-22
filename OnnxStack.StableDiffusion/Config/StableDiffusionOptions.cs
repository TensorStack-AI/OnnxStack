using System;

namespace OnnxStack.StableDiffusion.Config
{
    public class StableDiffusionOptions
    {
        private int _seed;

        /// <summary>
        /// Gets or sets the height.
        /// </summary>
        /// <value>
        ///  The height of the image. Default is 512 and must be a multiple of 8.
        /// </value>
        public int Height { get; set; } = 512;

        /// <summary>
        /// Gets or sets the width.
        /// </summary>
        /// <value>
        /// The width of the image. Default is 512 and must be a multiple of 8.
        /// </value>
        public int Width { get; set; } = 512;

        /// <summary>
        /// Gets or sets the number inference steps.
        /// </summary>
        /// <value>
        /// The number of steps to run inference for. The more steps the longer it will take to run the inference loop but the image quality should improve.
        /// </value>
        public int NumInferenceSteps { get; set; } = 15;

        /// <summary>
        /// Gets or sets the guidance scale.
        /// </summary>
        /// <value>
        /// The scale for the classifier-free guidance. The higher the number the more it will try to look like the prompt but the image quality may suffer.
        /// </value>
        public double GuidanceScale { get; set; } = 7.5;



        /// <summary>
        /// Gets or sets the seed.
        /// </summary>
        /// <value>
        /// If value is set to 0 a random seed is returned.
        /// </value>
        public int Seed
        {
            get
            { 
                if(_seed > 0)
                    return _seed;

                _seed = Random.Shared.Next();
                return _seed;
            }
            set { _seed = value; }
        }


        public string Prompt { get; set; }

        public string NegativePrompt { get; set; }

        public SchedulerType SchedulerType { get; set; }

        public void Initialize()
        {
           
        }


    }
}
