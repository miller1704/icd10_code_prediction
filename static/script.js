var quill = new Quill('#editor', {
              modules: {
                toolbar: [
                  [{ header: [1, 2, false] }],
                  ['bold', 'italic', 'underline'],
                  ['image', 'code-block']
                ]
              },
              placeholder: 'Type here...',
              theme: 'snow'  // or 'bubble'
            });


//initiate seconds counter to prevent over-querying
function currSecond() {
  return parseInt(Date().slice(22,24));
};

//define globals
let currentRequest = null;
let priorTime = currSecond();
let currTime = currSecond();

//define listener as
quill.on('text-change', function() {
  //setup real-time variables
  let input_length = quill.getLength()-1
  currTime = currSecond();
  //run prediction where 5 characters exist, 5 seconds has passed from prior run, and prior prediction has not been made
  if((input_length > 5) && (Math.abs(currTime - priorTime) >= 5) && (currentRequest == null)) {
    console.log('text valid for prediction run!')
    let text_start = Math.max(input_length - 200,0)
    let text = quill.getContents()['ops'][0]['insert'].slice(text_start, input_length)      
    currentRequest = $.ajax({
      dataType: "json"
      ,url: '/prediction'
      ,beforeSend : function(xhr, opts) {            
        priorTime = currTime;
      }
      ,data: {text: text}
      ,success : function(data) {
        $("#icd10_code").text(data.icd10_code);
        $("#icd10_description").text('Official Description: \n'+data.icd10_description);
        currentRequest = null;
      }
    });
  }
});




