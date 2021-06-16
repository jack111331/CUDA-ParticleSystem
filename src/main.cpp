#include "ParticleSystem.h"

using namespace std;

int main(int argc, char **argv) {
    if (argc < 3) {
        cerr << "You should specify the json config file, the command format should be..." << endl;
        cerr << "ParticleSystem.exe <config file> <output frame directory>" << endl;
        exit(1);
    }
    ParticleSystem *pSystem = new ParticleSystem();
    pSystem->m_outputDirectory = argv[2];
    pSystem->initialize(argv[1]);
    cout << "Initialized config" << endl;
    pSystem->simulation(0.05);
}