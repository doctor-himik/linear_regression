# linear_regression
sandbox for ML model testing and basic experiments

How to set up a virtual environment?
Creating a virtual environment for your project:
1. Install virtualenv. You need to have Python and the package manager pip installed. If you don't have them, first install 
   Python from the official site python.org, then install pip. To install virtualenv, run the command in the terminal: `pip install virtualenv`. 
2. Create a virtual environment. Navigate to your project directory and run: `virtualenv venv`. Here, `venv` is the name 
   of the directory where the virtual environment will be installed. For example, if your project is called "MyProject", 
   you can name the environment `myproject_env` and command will look like `virtualenv myproject_env`. 
3. Activate the virtual environment. In the command line, enter: `.\venv\Scripts\activate`. After activation, the command 
   prompt will show the prefix (venv), indicating that you are inside the virtual environment. 
4. To deactivate the virtual environment, run: deactivate. This will return you to the global Python environment so you 
   can continue working on other projects or tasks. 
5. On MacOS, run the following commands sequentially:
   - `virtualenv --python=python3 .venv`
   - then `source .venv/bin/activate`
   - then `pip install -r requirements.txt`. 
   - If you encounter issues, delete the virtual environment with `rm -rf .venv` and reinstall it.


# Problems & trouleshooting
 - Problem: fatal: unable to access 'https://github.com/xxx/xxx.git/': schannel: next InitializeSecurityContext failed: Unknown error (0x80092013)
 - Fix: `git config --global http.schannelCheckRevoke false`

 - Issue installing sklearn in venv - AttributeError: module 'pkgutil' has no attribute 'ImpImporter'. Did you mean: 'zipimporter'? 
 - You need to manually install pip for Python 3.12:
   - Step 1: `python -m ensurepip --upgrade`
   - Step 2: `python -m pip install --upgrade setuptools` 
   - Step 3: Instead of running pip install sklearn, use pip install scikit-learn. The 'sklearn' package on PyPI is deprecated;
     use 'scikit-learn' for pip commands.
