using System;

namespace OnnxStack.StableDiffusion.Helpers
{
    internal class MathHelpers
    {
        /// <summary>
        /// Approximation of the definite integral of an analytic smooth function on a closed interval.
        /// </summary>
        /// <param name="function">The analytic smooth function to integrate.</param>
        /// <param name="start">The start.</param>
        /// <param name="end">The end.</param>
        /// <param name="epsilon">The expected relative accuracy.</param>
        /// <returns></returns>
        public static float IntegrateOnClosedInterval(Func<double, double> function, double start, double end, double epsilon = 1e-4)
        {
            return (float)AdaptiveSimpson(function, start, end, epsilon);
        }


        /// <summary>
        /// Initializes the adaptive Simpson's rule by calculating initial values and calling the auxiliary method.
        /// </summary>
        /// <param name="f">The f.</param>
        /// <param name="a">a.</param>
        /// <param name="b">The b.</param>
        /// <param name="epsilon">The epsilon.</param>
        /// <returns></returns>
        private static double AdaptiveSimpson(Func<double, double> f, double a, double b, double epsilon)
        {
            double c = (a + b) / 2.0;
            double h = b - a;
            double fa = f(a);
            double fb = f(b);
            double fc = f(c);
            double s = (h / 6) * (fa + 4 * fc + fb);
            return AdaptiveSimpsonAux(f, a, b, epsilon, s, fa, fb, fc);
        }


        /// <summary>
        /// Recursively applies the Simpson's rule and adapts the interval size based on the estimated error.
        /// </summary>
        /// <param name="f">The f.</param>
        /// <param name="a">a.</param>
        /// <param name="b">The b.</param>
        /// <param name="epsilon">The epsilon.</param>
        /// <param name="s">The s.</param>
        /// <param name="fa">The fa.</param>
        /// <param name="fb">The fb.</param>
        /// <param name="fc">The fc.</param>
        /// <returns></returns>
        private static double AdaptiveSimpsonAux(Func<double, double> f, double a, double b, double epsilon, double s, double fa, double fb, double fc)
        {
            double c = (a + b) / 2.0;
            double h = b - a;
            double d = (a + c) / 2.0;
            double e = (c + b) / 2.0;
            double fd = f(d);
            double fe = f(e);
            double s1 = (h / 12) * (fa + 4 * fd + fc);
            double s2 = (h / 12) * (fc + 4 * fe + fb);
            double s_ = s1 + s2;
            if (Math.Abs(s_ - s) <= 15 * epsilon)
            {
                return s_ + (s_ - s) / 15.0;
            }
            return AdaptiveSimpsonAux(f, a, c, epsilon / 2, s1, fa, fc, fd) + AdaptiveSimpsonAux(f, c, b, epsilon / 2, s2, fc, fb, fe);
        }
    }
}
