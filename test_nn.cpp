//| This file is a part of the sferes2 framework.
//| Copyright 2016, ISIR / Universite Pierre et Marie Curie (UPMC)
//| Main contributor(s): Jean-Baptiste Mouret, mouret@isir.fr
//|
//| This software is a computer program whose purpose is to facilitate
//| experiments in evolutionary computation and evolutionary robotics.
//|
//| This software is governed by the CeCILL license under French law
//| and abiding by the rules of distribution of free software.  You
//| can use, modify and/ or redistribute the software under the terms
//| of the CeCILL license as circulated by CEA, CNRS and INRIA at the
//| following URL "http://www.cecill.info".
//|
//| As a counterpart to the access to the source code and rights to
//| copy, modify and redistribute granted by the license, users are
//| provided only with a limited warranty and the software's author,
//| the holder of the economic rights, and the successive licensors
//| have only limited liability.
//|
//| In this respect, the user's attention is drawn to the risks
//| associated with loading, using, modifying and/or developing or
//| reproducing the software by the user in light of its specific
//| status of free software, that may mean that it is complicated to
//| manipulate, and that also therefore means that it is reserved for
//| developers and experienced professionals having in-depth computer
//| knowledge. Users are therefore encouraged to load and test the
//| software's suitability as regards their requirements in conditions
//| enabling the security of their systems and/or data to be ensured
//| and, more generally, to use and operate it in the same conditions
//| as regards security.
//|
//| The fact that you are presently reading this means that you have
//| had knowledge of the CeCILL license and that you accept its terms.

#include <iostream>
#include <Eigen/Core>

#include <sferes/eval/parallel.hpp>
#include <sferes/gen/evo_float.hpp>
#include <sferes/modif/dummy.hpp>
#include <sferes/phen/parameters.hpp>
#include <sferes/run.hpp>
#include <sferes/stat/best_fit.hpp>
#include <sferes/stat/qd_container.hpp>
#include <sferes/stat/qd_selection.hpp>
#include <sferes/stat/qd_progress.hpp>


#include <sferes/fit/fit_qd.hpp>
#include <sferes/qd/container/archive.hpp>
#include <sferes/qd/container/kdtree_storage.hpp>
#include <sferes/qd/container/sort_based_storage.hpp>
#include <sferes/qd/container/grid.hpp>
#include <sferes/qd/quality_diversity.hpp>
#include <sferes/qd/selector/tournament.hpp>
#include <sferes/qd/selector/uniform.hpp>
#include <sferes/qd/selector/population_based.hpp>
#include <sferes/qd/selector/value_selector.hpp>



#include <boost/test/unit_test.hpp>

#include <modules/nn2/mlp.hpp>
  #include <modules/nn2/gen_dnn.hpp>
#include <modules/nn2/phen_dnn.hpp>

#include <modules/nn2/gen_dnn_ff.hpp>

#include <exp/exp_direct_target_encode/gen_dte.hpp>

#include <cmath>
#include <algorithm>


using namespace sferes;
using namespace sferes::gen::dnn;
using namespace sferes::gen::evo_float;

struct Params {
  struct evo_float {
    SFERES_CONST float mutation_rate = 0.3f;
    SFERES_CONST float cross_rate = 0.0f;
    SFERES_CONST mutation_t mutation_type = polynomial;
    SFERES_CONST cross_over_t cross_over_type = sbx;
    SFERES_CONST float eta_m = 10.0f;
    SFERES_CONST float eta_c = 10.0f;
  };

  struct parameters {
    // maximum value of parameters
    SFERES_CONST float min = -1.0f;
    // minimum value
    SFERES_CONST float max = 1.0f;
  };

  struct dnn {
    SFERES_CONST size_t nb_inputs = 2; //position of the robot gripper
    SFERES_CONST size_t nb_outputs  = 3; //angles of each arm
    SFERES_CONST size_t min_nb_neurons  = 4; 
    SFERES_CONST size_t max_nb_neurons  = 10;
    SFERES_CONST size_t min_nb_conns  = 20;
    SFERES_CONST size_t max_nb_conns  = 100;
    SFERES_CONST float  max_weight  = 2.0f;
    SFERES_CONST float  max_bias  = 2.0f;

    SFERES_CONST float m_rate_add_conn  = 1.0f;
    SFERES_CONST float m_rate_del_conn  = 1.0f;
    SFERES_CONST float m_rate_change_conn = 1.0f;
    SFERES_CONST float m_rate_add_neuron  = 1.0f;
    SFERES_CONST float m_rate_del_neuron  = 1.0f;

    SFERES_CONST int io_param_evolving = true;
    //SFERES_CONST init_t init = random_topology;
    SFERES_CONST init_t init = ff;
  };

    struct nov {
      SFERES_CONST size_t deep = 2;
      SFERES_CONST double l = 0.1; // TODO value ???
      SFERES_CONST double k = 25; // TODO right value?
      SFERES_CONST double eps = 0.1;// TODO right value??
  };

  // TODO: move to a qd::
  struct pop {
      // number of initial random points
      SFERES_CONST size_t init_size = 20; // nombre d'individus générés aléatoirement 
      SFERES_CONST size_t size = 20; // size of a batch
      SFERES_CONST size_t nb_gen = 1001; // nbr de gen pour laquelle l'algo va tourner 
      SFERES_CONST size_t dump_period = 200; 
  };

  struct qd {

      SFERES_CONST size_t dim = 3;
      SFERES_CONST size_t behav_dim = 3; //taille du behavior descriptor
      SFERES_ARRAY(size_t, grid_shape, 100, 100, 100);
  };
};


FIT_QD(nn_mlp){

  public :
    //Indiv : still do not know what it is 
    //IO : Neural Network Input and Output type
    template <typename Indiv>

      //void eval(Indiv & ind, IO & input, IO & target){ //ind : altered phenotype
      void eval(Indiv & ind){ //ind : altered phenotype

        //std::cout << "EVALUATION" <<std::endl;

        ind.nn().init(); //init neural network 


        //test mutations 
        std::vector<double> targ(3);
        targ = ind.gen().get_target();

        //std::cout << "target: " << targ[0] << " " << targ[1] << " " << targ[2] << std::endl;

        size_t nb_out = 3;
        //Eigen::Vector3d target = {0.77024555206298828, 1.3301527500152588, 1.5427114963531494}; //MLP's target
        Eigen::Vector3d target = {-0.43605732917785645, -0.40392923355102539, -0.29289793968200684};

        //Eigen::Vector3d pos = forward_model(target); //compute forward model to obtain gripper's position

        std::vector<float> in(2); //float or double?? 

        //in = {pos[0]*0.5 + 0.5, pos[1]*0.5 + 0.5}; //rescale position
        //in = {pos[0], pos[1]};
        in = {0.83299100918905145, 0.15455961613055846};

        //TODO : change it for an iteration of steps 

        for (int j = 0; j < ind.gen().get_depth() + 1; ++j)
                ind.nn().step(in);

        //ind.nn().step(in); //process a step with values contained in in 

        Eigen::Vector3d angles;

        angles[0] = 2*M_PI*(ind.nn().get_outf(0) - 0.5);
        angles[1] = 2*M_PI*(ind.nn().get_outf(1) - 0.5);
        angles[2] = 2*M_PI*(ind.nn().get_outf(2) - 0.5);


        double error  = - sqrt(square(target.array() - angles.array()).sum()); //-MSE


        this->_value = error; //compute -MSE as we intend to maximize a function

        std::vector<double> data = {angles[0], angles[1], angles[2]};

        this->set_desc(data); //Which behavior descriptor? The three motors angles 
      }
    

  Eigen::Vector3d forward_model(Eigen::VectorXd a){
    
    Eigen::VectorXd _l_arm=Eigen::VectorXd::Ones(a.size()+1);
    _l_arm(0)=0;
    _l_arm = _l_arm/_l_arm.sum();

    Eigen::Matrix4d mat=Eigen::Matrix4d::Identity(4,4);

    for(size_t i=0;i<a.size();i++){

      Eigen::Matrix4d submat;
      submat<<cos(a(i)), -sin(a(i)), 0, _l_arm(i), sin(a(i)), cos(a(i)), 0 , 0, 0, 0, 1, 0, 0, 0, 0, 1;
      mat=mat*submat;
    }
    
    Eigen::Matrix4d submat;
    submat<<1, 0, 0, _l_arm(a.size()), 0, 1, 0 , 0, 0, 0, 1, 0, 0, 0, 0, 1;
    mat=mat*submat;
    Eigen::VectorXd v=mat*Eigen::Vector4d(0,0,0,1);

    return v.head(3);

  }


};


int main(int argc, char **argv) 
{
    using namespace sferes;
    using namespace nn;

    std::cout << "STARTING INIT" <<std::endl;

    typedef nn_mlp<Params> fit_t; //TODO : Fitness to test

    typedef phen::Parameters<gen::EvoFloat<1, Params>, fit::FitDummy<>, Params> weight_t;
    typedef phen::Parameters<gen::EvoFloat<1, Params>, fit::FitDummy<>, Params> bias_t;
    typedef PfWSum<weight_t> pf_t;
    typedef AfSigmoidBias<bias_t> af_t;
    typedef sferes::gen::Dte<Neuron<pf_t, af_t>,  Connection<weight_t>, Params> gen_t; // TODO : change by DnnFF in order to use only feed-forward neural networks
                                                                                       // TODO : change by hyper NN in order to test hyper NEAT 

    //typedef sferes::gen::Dte<Neuron<pf_t, af_t>, Connection<weight_t>, Params> gen_t;

    typedef phen::Dnn<gen_t, fit_t, Params> phen_t;

    //start with a fit dummy for debugging
    // typedef phen::Parameters<gen::EvoFloat<1, Params>, fit::FitDummy<>, Params> weight_t;
    // typedef phen::Parameters<gen::EvoFloat<1, Params>, fit::FitDummy<>, Params> bias_t;
    // typedef PfWSum<weight_t> pf_t;
    // typedef AfTanh<bias_t> af_t;
    // typedef sferes::gen::Dnn<Neuron<pf_t, af_t>,  Connection<weight_t>, Params> gen_t; //does this exist in sferes??
    // typedef phen::Dnn<gen_t, fit::FitDummy<>, Params> phen_t;


    //typedef qd::selector::Uniform<phen_t, Params> select_t; //TODO : test other selector

    typedef qd::selector::getFitness ValueSelect_t;
    typedef qd::selector::Tournament<phen_t, ValueSelect_t, Params> select_t; 

    typedef qd::container::SortBasedStorage< boost::shared_ptr<phen_t> > storage_t; 
    typedef qd::container::Archive<phen_t, storage_t, Params> container_t; 

    //typedef eval::Eval<Params> eval_t; //(useful for debbuging)
    typedef eval::Parallel<Params> eval_t; //parallel eval (faster)
 
    typedef boost::fusion::vector< 
        stat::BestFit<phen_t, Params>, 
        stat::QdContainer<phen_t, Params>, 
        stat::QdProgress<phen_t, Params> 
        >
        stat_t; 

    typedef modif::Dummy<> modifier_t; //place holder
    
    typedef qd::QualityDiversity<phen_t, eval_t, stat_t, modifier_t, select_t, container_t, Params> qd_t; 
    //typedef qd::MapElites<phen_t, eval_t, stat_t, modifier_t, Params> qd_t;

    qd_t qd;
    //run_ea(argc, argv, qd); 

    std::cout << "INIT DONE...START RUN" << std::endl;

    qd.run();

    std::cout << "RUN DONE" << std::endl;

    std::cout<<"best fitness:" << qd.stat<0>().best()->fit().value() << std::endl;

    qd.stat<0>().best()->nn().init(); //init network for best fit

    std::vector<float> in(2); 
    in = {0.83299100918905145, 0.15455961613055846};

    for (int j = 0; j < qd.stat<0>().best()->gen().get_depth() + 1; ++j)
                qd.stat<0>().best()->nn().step(in);

    std::cout<<"best angles : " << 2*M_PI*(qd.stat<0>().best()->nn().get_outf(0) - 0.5) << " , " << 2*M_PI*(qd.stat<0>().best()->nn().get_outf(1) - 0.5) << " , " << 2*M_PI*(qd.stat<0>().best()->nn().get_outf(2) - 0.5) << std::endl;
    //std::cout<<"archive size:" << qd.stat<1>().archive().size() << std::endl;
    return 0;
    
}
