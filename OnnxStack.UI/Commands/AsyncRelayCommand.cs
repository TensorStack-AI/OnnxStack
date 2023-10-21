using System;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Input;

namespace OnnxStack.UI.Commands
{

    /// <summary>
    /// Async Relay command implemtation
    /// </summary>
    /// <seealso cref="System.Windows.Input.ICommand" />
    public class AsyncRelayCommand : ICommand
    {
        private readonly Func<Task> execute;
        private readonly Func<bool> canExecute;
        private long isExecuting;


        /// <summary>
        /// Initializes a new instance of the <see cref="AsyncRelayCommand"/> class.
        /// </summary>
        /// <param name="execute">The execute.</param>
        /// <param name="canExecute">The can execute.</param>
        public AsyncRelayCommand(Func<Task> execute, Func<bool> canExecute = null)
        {
            this.execute = execute;
            this.canExecute = canExecute ?? (() => true);
        }

        /// <summary>
        /// Occurs when changes occur that affect whether or not the command should execute.
        /// </summary>
        public event EventHandler CanExecuteChanged
        {
            add { CommandManager.RequerySuggested += value; }
            remove { CommandManager.RequerySuggested -= value; }
        }

        /// <summary>
        /// Raises the can execute changed.
        /// </summary>
        public void RaiseCanExecuteChanged()
        {
            CommandManager.InvalidateRequerySuggested();
        }

        /// <summary>
        /// Defines the method that determines whether the command can execute in its current state.
        /// </summary>
        /// <param name="parameter">Data used by the command.  If the command does not require data to be passed, this object can be set to <see langword="null" />.</param>
        /// <returns>
        ///   <see langword="true" /> if this command can be executed; otherwise, <see langword="false" />.
        /// </returns>
        public bool CanExecute(object parameter)
        {
            if (Interlocked.Read(ref isExecuting) != 0)
                return false;

            return canExecute();
        }


        /// <summary>
        /// Defines the method to be called when the command is invoked.
        /// </summary>
        /// <param name="parameter">Data used by the command.  If the command does not require data to be passed, this object can be set to <see langword="null" />.</param>
        public async void Execute(object parameter)
        {
            Interlocked.Exchange(ref isExecuting, 1);
            RaiseCanExecuteChanged();

            try
            {
                await execute();
            }
            finally
            {
                Interlocked.Exchange(ref isExecuting, 0);
                RaiseCanExecuteChanged();
            }
        }
    }

    /// <summary>
    /// Async Relay command with type argument implemtation
    /// </summary>
    /// <seealso cref="System.Windows.Input.ICommand" />
    public class AsyncRelayCommand<T> : ICommand
    {
        private readonly Func<T, Task> execute;
        private readonly Func<T, bool> canExecute;
        private long isExecuting;


        /// <summary>
        /// Initializes a new instance of the <see cref="AsyncRelayCommand{T}"/> class.
        /// </summary>
        /// <param name="execute">The execute.</param>
        /// <param name="canExecute">The can execute.</param>
        public AsyncRelayCommand(Func<T, Task> execute, Func<T, bool> canExecute = null)
        {
            this.execute = execute;
            this.canExecute = canExecute ?? (o => true);
        }

        /// <summary>
        /// Occurs when changes occur that affect whether or not the command should execute.
        /// </summary>
        public event EventHandler CanExecuteChanged
        {
            add { CommandManager.RequerySuggested += value; }
            remove { CommandManager.RequerySuggested -= value; }
        }

        /// <summary>
        /// Raises the can execute changed.
        /// </summary>
        public void RaiseCanExecuteChanged()
        {
            CommandManager.InvalidateRequerySuggested();
        }

        /// <summary>
        /// Defines the method that determines whether the command can execute in its current state.
        /// </summary>
        /// <param name="parameter">Data used by the command.  If the command does not require data to be passed, this object can be set to <see langword="null" />.</param>
        /// <returns>
        ///   <see langword="true" /> if this command can be executed; otherwise, <see langword="false" />.
        /// </returns>
        public bool CanExecute(object parameter)
        {
            if (Interlocked.Read(ref isExecuting) != 0)
                return false;

            return canExecute(parameter is T r ? r : default);
        }

        /// <summary>
        /// Defines the method to be called when the command is invoked.
        /// </summary>
        /// <param name="parameter">Data used by the command.  If the command does not require data to be passed, this object can be set to <see langword="null" />.</param>
        public async void Execute(object parameter)
        {
            Interlocked.Exchange(ref isExecuting, 1);
            RaiseCanExecuteChanged();

            try
            {
                await execute((T)parameter);
            }
            finally
            {
                Interlocked.Exchange(ref isExecuting, 0);
                RaiseCanExecuteChanged();
            }
        }
    }
}
