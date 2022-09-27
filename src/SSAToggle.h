#ifndef SSATOGGLE_H
#define SSATOGGLE_H

extern double e;

struct ToggleCell{
    int lineage;
    int parent;
    int rcdSize;
    double* time;
    int* green;
    int* red;

    ToggleCell(){
        this->lineage = 0;
        this->parent = 0;
        this->rcdSize = 0;
//        this->time = nullptr;
//        this->green = nullptr;
//        this->red = nullptr;
    }
    ~ToggleCell(){
        if(this->time){
            delete[] this->time;
            delete[] this->green;
            delete[] this->red;
            this->time = nullptr;
            this->green = nullptr;
            this->red = nullptr;
        }
    }

    ToggleCell& operator=(ToggleCell& cellTemp){
        lineage = cellTemp.lineage;
        parent = cellTemp.parent;
        rcdSize = cellTemp.rcdSize;
        if(cellTemp.rcdSize > 0){
            time = new double[rcdSize];
            green = new int[rcdSize];
            red = new int[rcdSize];
            for(int i=0; i<cellTemp.rcdSize; ++i){
                green[i] = cellTemp.green[i];
                red[i] = cellTemp.red[i];
                time[i] = cellTemp.time[i];
            }
        }
        return *this;
    };

};

struct cellBatch{
    ToggleCell* cells;
    int size;
    cellBatch(){
        this->cells = nullptr;
        this->size = 0;
    }
    ~cellBatch(){
        if(this->cells){
            delete[] this->cells;
            this->cells = nullptr;
        }
    }

};


int runSim(const double& gr, const int& green, const int& red,
           const double& endTime, const double& outputTime, const double& t0,
           double* saveT, int* saveX1, int* saveX2, int* saveSize);
int rumMultiSim(const int& threadNum, const double& gr, int* green, int* red,
                const double& endTime, const double& outputTime, int simsize, double* saveBuff, int* saveLength);
void runBatchSim(const int& threadNum, const double& gr, const int& green, const int& red,
                 const double& endTime, const double& outputTime, const int &maxCell,
                 ToggleCell** cellsarray, int* cellsSize);
void freeCellArray(ToggleCell* cell, int& size);
void freeCellMem(ToggleCell* cell);
#endif