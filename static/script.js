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




currentRequest = quill.on('text-change', function() {
  let currentRequest = null;
  let input_length = quill.getLength()-1
  if(input_length > 30) {
    let text_start = Math.max(input_length - 200,0)
    let text = quill.getContents()['ops'][0]['insert'].slice(text_start, input_length)
    let last_char = text[text.length-1]
    if (last_char == ' ') {         
       currentRequest = $.ajax({
          dataType: "json"
          ,url: '/prediction'
          ,beforeSend : function(xhr, opts) {if(currentRequest != null) {console.log('toomuch!'); xhr.abort();}}
          ,data: {text: text}
          ,success : function(data) {$("#icd10_code").text(data.icd10_code);$("#icd10_description").text('Official Description: \n'+data.icd10_description);}
        });
     }

  }
});




