<!DOCTYPE html>
<html>
<head>
<script src="https://cdn.jsdelivr.net/npm/chart.js@3.1.0/dist/chart.min.js"></script> 
<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.2/css/bootstrap.min.css">
<meta charset="UTF-8">
<title>Insert title here</title>
</head>
<body>
 {{data_male.confCase}} {{data_female.confCase}} {{data_00.confCase}} {{data_10.confCase}} {{data_20.confCase}} {{data_30.confCase}} {{data_40.confCase}} {{data_50.confCase}} {{data_60.confCase}} {{data_70.confCase}}  {{data_80.confCase}} 
	<div class="contaoiner">
		<div class="row">
			<div class="col-md=8">
				<canvas id="myChartPie"></canvas>
			</div>
		</div>
	</div>
	<div class="contaoiner">
		<div class="row">
			<div class="col-md=8">
				<canvas id="myChartPieDeath"></canvas>
			</div>
		</div>
	</div>
	<div class="contaoiner">
		<div class="row">
			<div class="col-md=8">
				<canvas id="myChartBarAge"></canvas>
			</div>
		</div>
	</div>
	<div class="contaoiner">
		<div class="row">
			<div class="col-md=8">
				<canvas id="myChartBarAgeDeath"></canvas>
			</div>
		</div>
	</div>
<script>
const label_age =['{{data_male.gubun}}','{{data_female.gubun}}'];
const label_gender = ['{{data_80.gubun}}', '{{data_70.gubun}}', '{{data_60.gubun}}', '{{data_50.gubun}}',
	'{{data_40.gubun}}', '{{data_30.gubun}}', '{{data_20.gubun}}', '{{data_10.gubun}}', '{{data_00.gubun}}'];

const data_gender= {
			  labels: label_age,
			  datasets: [
			    {
			      label: '남녀 확진자',
			      data: [{{data_male.confCase}}, {{data_female.confCase}}],
			      backgroundColor:[
			    	 'rgb(30, 144, 255, 0.4)',' rgb(255, 20, 147,0.4)'
			      ]
			    }
			  ]
			};
const data_gender_death= {
		  labels: label_age,
		  datasets: [
		    {
		      label: '남녀 확진자',
		      data: [{{data_male.death}}, {{data_female.death}}],
		      backgroundColor:[
		    	  'rgb(30, 144, 255, 0.4)',' rgb(255, 20, 147,0.4)'
		      ]
		    }
		  ]
		};
const data_age_death= {
		  labels: label_gender,
		  datasets: [
		    {
		      label: '연령별 확진자',
		      data: [{{data_80.confCase}}, {{data_70.confCase}}, {{data_60.confCase}}, {{data_50.confCase}},
		    		{{data_40.confCase}}, {{data_30.confCase}}, {{data_20.confCase}}, {{data_10.confCase}}, {{data_00.confCase}}],
		      backgroundColor:[
				'brick'
		      ]
		    }
		  ]
		};
const data_age= {
		  labels: label_gender,
		  datasets: [
		    {
		      label: '연령별 사망자',
		      data: [{{data_80.death}}, {{data_70.death}}, {{data_60.death}}, {{data_50.death}},
		    		{{data_40.death}}, {{data_30.death}}, {{data_20.death}}, {{data_10.death}}, {{data_00.death}}],
		      backgroundColor:[
				'brick'
		      ]
		    }
		  ]
		};
			let pieGenderConf = document.getElementById('myChartPie').getContext('2d');
			let pieGenderDeath = document.getElementById('myChartPieDeath').getContext('2d');
			let barAgeConf = document.getElementById('myChartBarAge').getContext('2d');
			let barAgeDeath = document.getElementById('myChartBarAgeDeath').getContext('2d');
			
			let pieChartOne = new Chart(pieGenderConf,{
			  type: 'pie',
			  data: data_gender,
			  options: {
			    responsive: true,
			    plugins: {
			      legend: {
			        position: 'top',
			      },
			      title: {
			        display: true,
			        text: '남녀 확진자'
			      }
			    }
			  },
			});
			
			let pieChartTwo = new Chart(pieGenderDeath,{
				  type: 'pie',
				  data: data_gender_death,
				  options: {
				    responsive: true,
				    plugins: {
				      legend: {
				        position: 'top',
				      },
				      title: {
				        display: true,
				        text: '남녀 사망자'
				      }
				    }
				  },
				});
			let barAgeConfOne = new Chart(barAgeConf,{	
				type : 'bar',	
				data: data_age_death,
				options: {
					indexAxis: 'y',
				    plugins: {
				      title: {
				        display: true,
				        text: '연령별 확진자',
				      },
				    },
				    responsive: true,
				    scales: {
				      x: {
				        stacked: true,
				      },
				      y: {
				        stacked: true,
				      }
				    }
				}
			});
			let barAgeDeathOne = new Chart(barAgeDeath,{	
				type : 'bar',	
				data: data_age,
				options: {
					indexAxis: 'y',
				    plugins: {
				      title: {
				        display: true,
				        text: '연령별 사망자',
				      },
				    },
				    responsive: true,
				    scales: {
				      x: {
				        stacked: true,
				      },
				      y: {
				        stacked: true,
				      }
				    }
				}
			});

</script>


</body>
</html>