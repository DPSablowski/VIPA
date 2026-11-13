"""Starts the VIPA finesse calculator:  python start_calculator.py
Self-test without GUI:                python start_calculator.py --test"""
import multiprocessing as mp, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "kernel"))

if __name__ == "__main__":
    mp.freeze_support()
    if "--test" in sys.argv:
        import test_calculator; sys.exit(test_calculator.main())
    import calculator_gui; calculator_gui.main()
